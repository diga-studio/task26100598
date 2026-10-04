package itemcf

import (
	"math"
	"slices"
)

// Recommend returns up to 20 recommendations for a user of the snapshot,
// ordered by score desc, item ID asc. Blacklisted items are excluded from
// candidates. Unknown users get nil.
func (r *Recommender) Recommend(userID int, blacklist []int) []Recommendation {
	u, ok := r.userIdx[userID]
	if !ok {
		return nil
	}
	q := newQuery(r, blacklist)
	hist := r.uItems[r.userOff[u]:r.userOff[u+1]]
	sumAbs := r.sumAbs[u]

	q.accumulate(r, u, hist, sumAbs)
	top := q.rank(r, sumAbs)
	out := make([]Recommendation, len(top))
	for i, c := range top {
		out[i] = q.explain(r, hist, sumAbs, c)
	}
	return out
}

// query holds per-call scratch buffers.
type query struct {
	inHist  []bool    // dense item -> rated by query user
	blocked []bool    // dense item -> blacklisted
	sUser   []float64 // dense user -> s_v accumulator
	seenU   []bool    // dense user -> already appended to touchU
	touchU  []int32   // users with nonzero s_v
	scores  []float64 // dense item -> unnormalized score accumulator
	seenI   []bool    // dense item -> already appended to touchI
	touchI  []int32   // items touched by scoring
	cooc    []float64 // dense item -> co-occurrence accumulator for explain
}

func newQuery(r *Recommender, blacklist []int) *query {
	q := &query{
		inHist:  make([]bool, len(r.itemIDs)),
		blocked: make([]bool, len(r.itemIDs)),
		sUser:   make([]float64, len(r.userIDs)),
		seenU:   make([]bool, len(r.userIDs)),
		scores:  make([]float64, len(r.itemIDs)),
		seenI:   make([]bool, len(r.itemIDs)),
		cooc:    make([]float64, len(r.itemIDs)),
	}
	for _, id := range blacklist {
		if it, ok := r.itemIdx[id]; ok {
			q.blocked[it] = true
		}
	}
	return q
}

// accumulate computes unnormalized scores: scores[c] = sum_v w_vc * s_v
// with s_v = sum_h w_h * w_vh / norm_h over the query user's history.
func (q *query) accumulate(r *Recommender, u int32, hist []uItem, sumAbs float64) {
	for _, h := range hist {
		q.inHist[h.item] = true
	}
	if sumAbs == 0 {
		return
	}
	for _, h := range hist {
		norm := r.itemNorm[h.item]
		if norm == 0 {
			continue
		}
		f := h.w / norm
		for p := r.itemOff[h.item]; p < r.itemOff[h.item+1]; p++ {
			ra := r.iRaters[p]
			if !q.seenU[ra.user] {
				q.seenU[ra.user] = true
				q.touchU = append(q.touchU, ra.user)
			}
			q.sUser[ra.user] += f * ra.w
		}
	}
	for _, v := range q.touchU {
		sv := q.sUser[v]
		if sv == 0 {
			continue
		}
		for k := r.userOff[v]; k < r.userOff[v+1]; k++ {
			ui := r.uItems[k]
			if !q.seenI[ui.item] {
				q.seenI[ui.item] = true
				q.touchI = append(q.touchI, ui.item)
			}
			q.scores[ui.item] += ui.w * sv
		}
	}
}

// finalScore normalizes the accumulated score of item c.
func (r *Recommender) finalScore(raw float64, c int32, sumAbs float64) float64 {
	if sumAbs == 0 || r.itemNorm[c] == 0 {
		return 0
	}
	return raw / (r.itemNorm[c] * sumAbs * math.Log(2+float64(r.itemPop[c])))
}

// rank selects the top-20 candidate items by score desc, item ID asc.
func (q *query) rank(r *Recommender, sumAbs float64) []int32 {
	cands := make([]int32, 0, len(r.itemIDs))
	for c := int32(0); c < int32(len(r.itemIDs)); c++ {
		if !q.inHist[c] && !q.blocked[c] {
			cands = append(cands, c)
		}
	}
	slices.SortFunc(cands, func(a, b int32) int {
		sa := r.finalScore(q.scores[a], a, sumAbs)
		sb := r.finalScore(q.scores[b], b, sumAbs)
		if sa != sb {
			if sa > sb {
				return -1
			}
			return 1
		}
		ia, ib := r.itemIDs[a], r.itemIDs[b]
		if ia != ib {
			return ia - ib
		}
		return 0
	})
	if len(cands) > 20 {
		cands = cands[:20]
	}
	return cands
}

// explain computes per-history-item contributions for candidate c and
// packs the top-3 (by |value| desc, item ID asc) plus the rest sum.
// Score is rebuilt as topSum + rest so the parts restore it exactly.
func (q *query) explain(r *Recommender, hist []uItem, sumAbs float64, c int32) Recommendation {
	var invSumAbs float64
	if sumAbs != 0 {
		invSumAbs = 1 / sumAbs
	}
	normC := r.itemNorm[c]
	logPop := math.Log(2 + float64(r.itemPop[c]))

	// cooc[h] = sum over raters v of c with h in I_v of w_vh * w_vc.
	var touchList []int32
	for p := r.itemOff[c]; p < r.itemOff[c+1]; p++ {
		ra := r.iRaters[p]
		for k := r.userOff[ra.user]; k < r.userOff[ra.user+1]; k++ {
			ui := r.uItems[k]
			if q.inHist[ui.item] {
				if q.cooc[ui.item] == 0 {
					touchList = append(touchList, ui.item)
				}
				q.cooc[ui.item] += ui.w * ra.w
			}
		}
	}

	contribs := make([]Contribution, len(hist))
	for i, h := range hist {
		var v float64
		if normC != 0 && r.itemNorm[h.item] != 0 {
			v = h.w * q.cooc[h.item] / (r.itemNorm[h.item] * normC) * invSumAbs / logPop
		}
		contribs[i] = Contribution{ItemID: r.itemIDs[h.item], Value: v}
	}
	for _, it := range touchList {
		q.cooc[it] = 0
	}

	order := make([]int, len(contribs))
	for i := range order {
		order[i] = i
	}
	slices.SortFunc(order, func(a, b int) int {
		aa, bb := math.Abs(contribs[a].Value), math.Abs(contribs[b].Value)
		if aa != bb {
			if aa > bb {
				return -1
			}
			return 1
		}
		return contribs[a].ItemID - contribs[b].ItemID
	})
	nTop := 3
	if len(order) < nTop {
		nTop = len(order)
	}
	topSet := make([]int, nTop)
	inTop := make([]bool, len(contribs))
	for i := 0; i < nTop; i++ {
		topSet[i] = order[i]
		inTop[order[i]] = true
	}
	rec := Recommendation{ItemID: r.itemIDs[c], Top: make([]Contribution, nTop)}
	var topSum float64
	for i, idx := range topSet {
		rec.Top[i] = contribs[idx]
		topSum += contribs[idx].Value
	}
	var rest float64
	for i, ct := range contribs {
		if !inTop[i] {
			rest += ct.Value
		}
	}
	rec.Rest = rest
	rec.Score = topSum + rest
	return rec
}
