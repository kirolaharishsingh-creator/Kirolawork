# Split the Tripo chair (one mesh, ~1500 loose shells) into exploded-view parts.
# Each loose shell goes wholly to one part, chosen by its centroid (x forward, y lateral, z up, metres).
import numpy as np

PART_NAMES = ['headrest', 'backrest', 'frame', 'lumbar', 'seat', 'arm_r', 'arm_l',
              'mechanism', 'gas_lift', 'base', 'wheel0', 'wheel1', 'wheel2', 'wheel3', 'wheel4']
HUB = np.array([0.01, 0.0])

def classify(c):
    x, y, z = c
    if z > 0.385 or (x < -0.275 and z > 0.26): return 'headrest'
    if z > -0.2 and (abs(y) > 0.212 or (z > -0.06 and abs(y) > 0.17 and x > -0.12)):
        return 'arm_r' if y > 0 else 'arm_l'
    if z > 0.09: return 'frame' if (x < -0.215 and abs(y) < 0.13) else 'backrest'
    if z > -0.06: return 'frame' if x < -0.215 else 'lumbar'
    if z > -0.17: return 'frame' if x < -0.17 else 'seat'
    if z > -0.255: return 'frame' if x < -0.12 else 'mechanism'
    r = np.hypot(x - HUB[0], y - HUB[1])
    if z > -0.345 and r < 0.06: return 'gas_lift'
    if z > -0.255 - 0.0 and r >= 0.06: return 'mechanism'
    if z < -0.415 and r > 0.1:
        return 'wheel'
    return 'base'

def component_adjacency(co, lab, tri, r=5e-4):
    # shells touch along texture seams: open-boundary vertices of different shells that coincide
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    e = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), 1)
    ue, cnt = np.unique(e, axis=0, return_counts=True)
    bv = np.unique(ue[cnt == 1])
    pairs = cKDTree(co[bv]).query_pairs(r, output_type='ndarray')
    a, b = lab[bv[pairs[:, 0]]], lab[bv[pairs[:, 1]]]
    k = a != b; n = lab.max() + 1
    W = coo_matrix((np.ones(k.sum()), (a[k], b[k])), shape=(n, n)).tocsr()
    return W + W.T

def smooth_parts(cpart, W, cnt, rounds=4):
    # every part must be one connected piece: stray groups of shells move to the part they are attached to
    from scipy.sparse.csgraph import connected_components
    cpart = cpart.copy()
    for _ in range(rounds):
        moved = 0
        for p in range(len(PART_NAMES)):
            idx = np.nonzero(cpart == p)[0]
            if len(idx) < 2: continue
            ng, g = connected_components(W[idx][:, idx], directed=False)
            if ng == 1: continue
            size = np.bincount(g, cnt[idx]); keep = np.argmax(size)
            for gi in range(ng):
                if gi == keep: continue
                comps = idx[g == gi]
                w = np.asarray(W[comps].sum(0)).ravel()
                votes = np.bincount(cpart, w, minlength=len(PART_NAMES)); votes[p] = 0
                if votes.max() > 0: cpart[comps] = np.argmax(votes); moved += 1
        if not moved: break
    return cpart

def label_components(co, lab, tri=None):
    n = lab.max() + 1
    cnt = np.bincount(lab, minlength=n)
    cen = np.stack([np.bincount(lab, co[:, i], minlength=n) for i in range(3)], 1) / np.maximum(cnt, 1)[:, None]
    names = [classify(c) for c in cen]
    # the five casters: cluster wheel shells by angle around the hub (split at the 5 widest gaps)
    wi = [i for i, nm in enumerate(names) if nm == 'wheel']
    ang = np.degrees(np.arctan2(cen[wi, 1] - HUB[1], cen[wi, 0] - HUB[0])) % 360
    o = np.argsort(ang); a = ang[o]
    gaps = np.diff(np.r_[a, a[0] + 360]); cuts = np.sort(np.argsort(-gaps)[:5])
    grp = np.zeros(len(a), int)
    for g, (s0, s1) in enumerate(zip(cuts, np.r_[cuts[1:], cuts[0] + len(a)])):
        for k in range(s0 + 1, s1 + 1): grp[k % len(a)] = g
    for k, idx in enumerate(o): names[wi[idx]] = 'wheel%d' % grp[k]
    cpart = np.array([PART_NAMES.index(nm) for nm in names])
    if tri is not None:
        cpart = smooth_parts(cpart, component_adjacency(co, lab, tri), cnt)
    return cpart
