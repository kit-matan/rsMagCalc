//! The ground-state guard on large sparse cells: is H2(q) + shift I positive
//! definite? -- decided by a SPARSE Cholesky (multifrontal, minimum-degree
//! ordering) instead of one dense `zpotrf` of the whole 2D x 2D matrix.
//!
//! It decides the SAME question on the SAME matrix, exactly:
//!   * the matrix entries are bit-identical to the dense path's: `SunModel::
//!     assemble` emits its (row, col, value) contributions in a q-independent
//!     order, the analysis records which slot each one lands in, and the per-q
//!     build sums the same values in the same order; the symmetrisation
//!     0.5 (H2[r, c] + conj H2[c, r]) and the diagonal shift are `h2_colmajor`'s;
//!   * a symmetric permutation P^T A P is positive definite iff A is, and the
//!     multifrontal factorization computes exactly the pivots of the Cholesky of
//!     P^T A P (every pivot block is a diagonal block of a Schur complement), so
//!     the first front whose `zpotrf` fails is a non-positive pivot of that
//!     Cholesky, and if none fails the factor exists. Only the rounding differs
//!     from the dense factorization, i.e. a verdict can move only when
//!     min eig (H2 + shift I) is itself at round-off -- which is what the shift
//!     (h2_rel_tolerance x the reference scale, 1e-6 relative) is there to
//!     prevent. No Lanczos / Ritz bound: those err in the unsafe direction.
//!
//! Analysis (once per packed model, lazily): the slot sequence; the symmetric
//! adjacency of H2; indistinguishable variables (identical closed neighbourhoods
//! -- the Nambu pair and the M bosons of a site) compressed into supervariables;
//! a weighted minimum-degree elimination on the compressed graph, whose cliques
//! ARE the factor's column structures; chains merged into supernodes when no fill
//! is added; then per front its index list, its slice of the original entries
//! and the relative positions of each child's update rows. Every relative index
//! is looked up, never assumed, so a structure the fronts cannot hold is an
//! analysis error (and the dense path answers), not a silent drop.
//!
//! Per q: assemble the slots, then the fronts in order -- scatter the original
//! entries, extend-add the children, `zpotrf` the pivot block (stop at the first
//! failure), `ztrsm` the off-diagonal block, `zherk` the Schur update.
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use numpy::Complex64;

use crate::lapack;
use crate::sun_lswt::SunModel;

type C = Complex64;

const ZERO: C = C { re: 0.0, im: 0.0 };
const NONE: u32 = u32::MAX;

struct Front {
    /// Front order m = pivots + update rows.
    m: usize,
    /// Pivot count.
    np: usize,
    /// (child front, position in THIS front of each of the child's update rows).
    children: Vec<(u32, Vec<u32>)>,
    /// Off-diagonal original entries: (linear index row + col m in the front,
    /// slot of H2[row, col], slot of H2[col, row]) -- either slot may be NONE.
    entries: Vec<(u32, u32, u32)>,
    /// Diagonal original entries: (linear index, slot or NONE).
    diag: Vec<(u32, u32)>,
}

/// A packed model's sparse-guard analysis.
pub(crate) struct SparseGuard {
    seq: Vec<u32>,
    n_slots: usize,
    fronts: Vec<Front>,
    pub(crate) n_super: usize,
    pub(crate) nnz_l: usize,
    pub(crate) flops: f64,
    pub(crate) max_front: usize,
    /// Peak bytes of front buffers alive at once in one factorization.
    pub(crate) peak_bytes: usize,
}

/// Merge two sorted, duplicate-free lists into `out`, skipping `skip_a` and `skip_b`.
fn merge_into(a: &[u32], b: &[u32], skip_a: u32, skip_b: u32, out: &mut Vec<u32>) {
    out.clear();
    let (mut i, mut j) = (0, 0);
    while i < a.len() || j < b.len() {
        let x = if j == b.len() || (i < a.len() && a[i] < b[j]) {
            i += 1;
            a[i - 1]
        } else if i == a.len() || b[j] < a[i] {
            j += 1;
            b[j - 1]
        } else {
            i += 1;
            j += 1;
            a[i - 1]
        };
        if x != skip_a && x != skip_b {
            out.push(x);
        }
    }
}

impl SparseGuard {
    pub(crate) fn analyse(model: &SunModel) -> Result<SparseGuard, String> {
        let n = model.n;
        // 1. the slot of each contribution, in emission order
        let mut slot_of: HashMap<(u32, u32), u32> = HashMap::new();
        let mut rc: Vec<(u32, u32)> = Vec::new();
        let mut seq: Vec<u32> = Vec::new();
        model.assemble([0.0; 3], |r, c, _| {
            let key = (r as u32, c as u32);
            let s = *slot_of.entry(key).or_insert_with(|| {
                rc.push(key);
                (rc.len() - 1) as u32
            });
            seq.push(s);
        });
        let n_slots = rc.len();

        // 2. symmetric adjacency (no diagonal)
        let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
        for &(r, c) in &rc {
            if r != c {
                adj[r as usize].push(c);
                adj[c as usize].push(r);
            }
        }
        for a in adj.iter_mut() {
            a.sort_unstable();
            a.dedup();
        }

        // 3. supervariables: identical closed neighbourhoods
        let mut sv_of = vec![0u32; n];
        let mut members: Vec<Vec<u32>> = Vec::new();
        {
            let mut by_nbhd: HashMap<Vec<u32>, u32> = HashMap::new();
            for v in 0..n {
                let mut closed = adj[v].clone();
                let at = closed.partition_point(|&x| x < v as u32);
                closed.insert(at, v as u32);
                let id = *by_nbhd.entry(closed).or_insert_with(|| {
                    members.push(Vec::new());
                    (members.len() - 1) as u32
                });
                sv_of[v] = id;
                members[id as usize].push(v as u32);
            }
        }
        let ns = members.len();
        let w: Vec<usize> = members.iter().map(|m| m.len()).collect();
        let mut sadj: Vec<Vec<u32>> = vec![Vec::new(); ns];
        for v in 0..n {
            let s = sv_of[v];
            for &u in &adj[v] {
                let t = sv_of[u as usize];
                if t != s {
                    sadj[s as usize].push(t);
                }
            }
        }
        for a in sadj.iter_mut() {
            a.sort_unstable();
            a.dedup();
        }
        drop(adj);

        // 4. weighted minimum degree on the compressed graph (explicit
        //    elimination graph; ties to the lowest index, so it is deterministic)
        let wsum = |l: &[u32]| l.iter().map(|&u| w[u as usize]).sum::<usize>();
        let mut deg: Vec<usize> = sadj.iter().map(|a| wsum(a)).collect();
        let mut heap: BinaryHeap<Reverse<(usize, u32)>> =
            (0..ns).map(|v| Reverse((deg[v], v as u32))).collect();
        let mut gone = vec![false; ns];
        let mut step = vec![0u32; ns];
        let mut order: Vec<u32> = Vec::with_capacity(ns);
        let mut clique: Vec<Vec<u32>> = vec![Vec::new(); ns];
        let mut buf: Vec<u32> = Vec::new();
        while let Some(Reverse((dg, v))) = heap.pop() {
            let vi = v as usize;
            if gone[vi] || dg != deg[vi] {
                continue;
            }
            gone[vi] = true;
            step[vi] = order.len() as u32;
            order.push(v);
            let nb = std::mem::take(&mut sadj[vi]);
            for &u in &nb {
                let ui = u as usize;
                merge_into(&sadj[ui], &nb, u, v, &mut buf);
                std::mem::swap(&mut sadj[ui], &mut buf);
                deg[ui] = wsum(&sadj[ui]);
                heap.push(Reverse((deg[ui], u)));
            }
            clique[vi] = nb;
        }
        if order.len() != ns {
            return Err(format!("minimum degree eliminated {} of {ns} supervariables", order.len()));
        }

        // 5. fronts: each supervariable's clique is its update set; merge v into
        //    a child front whose update set is exactly {v} + clique(v) (no fill)
        let mut child_fronts: Vec<Vec<u32>> = vec![Vec::new(); ns]; // by parent sv
        let mut f_piv: Vec<Vec<u32>> = Vec::new(); // supervariables, in order
        let mut f_upd: Vec<u32> = Vec::new(); // the sv whose clique is the update set
        let mut f_kids: Vec<Vec<u32>> = Vec::new();
        for &v in &order {
            let vi = v as usize;
            let want = w[vi] + wsum(&clique[vi]);
            let kids = std::mem::take(&mut child_fronts[vi]);
            let merge = kids.iter().position(|&f| wsum(&clique[f_upd[f as usize] as usize]) == want);
            let f = match merge {
                Some(k) => {
                    let f = kids[k];
                    f_piv[f as usize].push(v);
                    f_upd[f as usize] = v;
                    f
                }
                None => {
                    f_piv.push(vec![v]);
                    f_upd.push(v);
                    f_kids.push(Vec::new());
                    (f_piv.len() - 1) as u32
                }
            };
            for (k, &c) in kids.iter().enumerate() {
                if Some(k) != merge {
                    f_kids[f as usize].push(c);
                }
            }
            if let Some(&p) = clique[vi].iter().min_by_key(|&&u| step[u as usize]) {
                child_fronts[p as usize].push(f);
            }
        }
        // processing order: by the elimination step of the LAST pivot
        let nf = f_piv.len();
        let mut forder: Vec<u32> = (0..nf as u32).collect();
        forder.sort_by_key(|&f| step[*f_piv[f as usize].last().unwrap() as usize]);
        let mut new_id = vec![0u32; nf];
        for (k, &f) in forder.iter().enumerate() {
            new_id[f as usize] = k as u32;
        }
        // global position of every variable: front pivots concatenated
        let mut pos = vec![0u32; n];
        let mut p = 0u32;
        for &f in &forder {
            for &s in &f_piv[f as usize] {
                for &x in &members[s as usize] {
                    pos[x as usize] = p;
                    p += 1;
                }
            }
        }

        // 6. per front: index list, children's relative maps, original entries
        let mut idx_lists: Vec<Vec<u32>> = Vec::with_capacity(nf);
        let mut fronts: Vec<Front> = Vec::with_capacity(nf);
        let mut front_of_var = vec![0u32; n];
        for (k, &f) in forder.iter().enumerate() {
            let fu = f as usize;
            let mut piv: Vec<u32> = f_piv[fu].iter()
                .flat_map(|&s| members[s as usize].iter().copied()).collect();
            piv.sort_by_key(|&x| pos[x as usize]);
            let mut upd: Vec<u32> = clique[f_upd[fu] as usize].iter()
                .flat_map(|&s| members[s as usize].iter().copied()).collect();
            upd.sort_by_key(|&x| pos[x as usize]);
            if let (Some(&a), Some(&b)) = (piv.last(), upd.first()) {
                if pos[b as usize] <= pos[a as usize] {
                    return Err(format!("front {k}: an update row precedes a pivot"));
                }
            }
            for &x in &piv {
                front_of_var[x as usize] = k as u32;
            }
            let np = piv.len();
            piv.extend_from_slice(&upd);
            fronts.push(Front { m: piv.len(), np, children: Vec::new(), entries: Vec::new(),
                                diag: Vec::new() });
            idx_lists.push(piv);
        }
        // local position lookup via a scratch map
        let mut local = vec![NONE; n];
        for k in 0..nf {
            for (i, &x) in idx_lists[k].iter().enumerate() {
                local[x as usize] = i as u32;
            }
            let f = forder[k] as usize;
            let mut children = Vec::new();
            for &c in &f_kids[f] {
                let cid = new_id[c as usize] as usize;
                if cid >= k {
                    return Err(format!("front {cid} is a child of front {k} but comes after it"));
                }
                let cl = &idx_lists[cid];
                let cnp = fronts[cid].np;
                let mut rel = Vec::with_capacity(cl.len() - cnp);
                for &x in &cl[cnp..] {
                    let l = local[x as usize];
                    if l == NONE {
                        return Err(format!("update row {x} of front {cid} is not in its parent {k}"));
                    }
                    if let Some(&prev) = rel.last() {
                        if l <= prev {
                            return Err(format!("front {k}: child {cid}'s rows are out of order"));
                        }
                    }
                    rel.push(l);
                }
                children.push((cid as u32, rel));
            }
            fronts[k].children = children;
            for &x in &idx_lists[k] {
                local[x as usize] = NONE;
            }
        }
        // every child must be claimed by exactly one parent
        let mut claimed = vec![false; nf];
        for fr in &fronts {
            for (c, _) in &fr.children {
                if std::mem::replace(&mut claimed[*c as usize], true) {
                    return Err(format!("front {c} has two parents"));
                }
            }
        }
        for k in 0..nf {
            if fronts[k].m > fronts[k].np && !claimed[k] {
                return Err(format!("front {k} has an update block and no parent"));
            }
        }
        // original entries: each structural (r, c) of H2 goes to the front of
        // whichever variable comes first, as (later row, earlier column)
        let mut pairs: HashMap<(u32, u32), (u32, u32)> = HashMap::new();
        for (s, &(r, c)) in rc.iter().enumerate() {
            let (hi, lo, fwd) = if pos[r as usize] >= pos[c as usize] { (r, c, true) } else { (c, r, false) };
            let e = pairs.entry((hi, lo)).or_insert((NONE, NONE));
            if fwd { e.0 = s as u32 } else { e.1 = s as u32 }
        }
        // H2[hi, lo] = 0.5 (raw[hi, lo] + conj raw[lo, hi]); the diagonal always
        // exists (the shift)
        for x in 0..n as u32 {
            pairs.entry((x, x)).or_insert((NONE, NONE));
        }
        let mut keys: Vec<(u32, u32)> = pairs.keys().copied().collect();
        keys.sort_unstable();
        for (hi, lo) in keys {
            let (a, b) = pairs[&(hi, lo)];
            let k = front_of_var[lo as usize] as usize;
            let lst = &idx_lists[k];
            let fr = &mut fronts[k];
            let col = lst[..fr.np].iter().position(|&x| x == lo)
                .ok_or_else(|| format!("variable {lo} is not a pivot of its front"))?;
            let row = lst.iter().position(|&x| x == hi)
                .ok_or_else(|| format!("entry ({hi}, {lo}) falls outside front {k}"))?;
            let lin = (row + col * fr.m) as u32;
            if hi == lo {
                // a diagonal contribution is recorded as (x, x) with fwd = true
                fr.diag.push((lin, a));
            } else {
                fr.entries.push((lin, a, b));
            }
        }

        // 7. cost model and peak memory (children freed when their parent runs)
        let (mut nnz_l, mut flops, mut max_front) = (0usize, 0.0f64, 0usize);
        let (mut live, mut peak) = (0usize, 0usize);
        for fr in &fronts {
            let (np, nu) = (fr.np as f64, (fr.m - fr.np) as f64);
            nnz_l += fr.np * (fr.np + 1) / 2 + fr.np * (fr.m - fr.np);
            flops += np * np * np / 3.0 + np * np * nu + np * nu * nu;
            max_front = max_front.max(fr.m);
            live += fr.m * fr.m * 16;
            peak = peak.max(live);
            for (c, _) in &fr.children {
                let cm = fronts[*c as usize].m;
                live -= cm * cm * 16;
            }
            if fr.m == fr.np {
                live -= fr.m * fr.m * 16;
            }
        }
        Ok(SparseGuard { seq, n_slots, fronts, n_super: ns, nnz_l, flops, max_front,
                         peak_bytes: peak })
    }

    /// Per-q bytes outside the fronts (the assembled slots).
    pub(crate) fn n_slots_bytes(&self) -> usize {
        self.n_slots * 16
    }

    /// Number of fronts (supernodes).
    pub(crate) fn n_fronts(&self) -> usize {
        self.fronts.len()
    }

    /// Is H2(q) + shift I positive definite?
    pub(crate) fn positive_definite(&self, model: &SunModel, q: [f64; 3], shift: f64) -> bool {
        let mut raw = vec![ZERO; self.n_slots];
        let mut k = 0usize;
        model.assemble(q, |_, _, v| {
            raw[self.seq[k] as usize] += v;
            k += 1;
        });
        debug_assert_eq!(k, self.seq.len());
        let get = |s: u32| if s == NONE { ZERO } else { raw[s as usize] };
        let mut bufs: Vec<Option<Vec<C>>> = (0..self.fronts.len()).map(|_| None).collect();
        for (fi, fr) in self.fronts.iter().enumerate() {
            let (m, np) = (fr.m, fr.np);
            let nu = m - np;
            let mut f = vec![ZERO; m * m];
            for &(lin, s) in &fr.diag {
                let a = get(s);
                let mut v = 0.5 * (a + a.conj());
                v.re += shift;
                f[lin as usize] = v;
            }
            for &(lin, a, b) in &fr.entries {
                f[lin as usize] = 0.5 * (get(a) + get(b).conj());
            }
            for (c, rel) in &fr.children {
                let u = bufs[*c as usize].take().expect("child front factored first");
                let cf = &self.fronts[*c as usize];
                let (cm, cnp) = (cf.m, cf.np);
                for (jj, &cj) in rel.iter().enumerate() {
                    let src = &u[(cnp + jj) * cm + cnp..(cnp + jj + 1) * cm];
                    let dst = cj as usize * m;
                    for (ii, &ci) in rel.iter().enumerate().skip(jj) {
                        f[dst + ci as usize] += src[ii]; // child row cnp + ii
                    }
                }
            }
            if lapack::zpotrf_lower_ld(np, &mut f, m).is_err() {
                return false;
            }
            if nu > 0 {
                let p = f.as_mut_ptr();
                // F21 <- F21 L11^{-H};  F22 <- F22 - F21 F21^H
                unsafe {
                    lapack::ztrsm_right_lower_conj(nu, np, p, m, p.add(np), m);
                    lapack::zherk_lower_sub(nu, np, p.add(np), m, p.add(np * m + np), m);
                }
                bufs[fi] = Some(f);
            }
        }
        true
    }
}
