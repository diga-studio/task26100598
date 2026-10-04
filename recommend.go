package itemcf

import (
	"math"
	"sort"
)

const topN = 20
const topContrib = 3

// Recommend 为快照中的用户生成推荐，blacklist 中的商品被排除出候选。
// 返回按得分降序、商品ID升序的前 20 件商品，不足 20 件则全部返回。
// 暂存空间在多次调用间复用，因此不支持并发调用。
func (r *Recommender) Recommend(userID int, blacklist []int) []Recommendation {
	ui, ok := r.userIndex[userID]
	if !ok {
		return nil
	}
	history := r.byUser[ui]
	r.buf.lazyInit(len(r.userIDs), len(r.itemIDs))
	b := &r.buf

	// 用户历史：rated 标记与权重查表；候选排除集额外并入黑名单。
	sumAbs := 0.0
	for _, e := range history {
		b.rated[e.itemIdx] = true
		b.histW[e.itemIdx] = e.weight
		b.excluded[e.itemIdx] = true
		sumAbs += math.Abs(e.weight)
	}
	blackIdx := make([]int, 0, len(blacklist))
	for _, id := range blacklist {
		if ii, ok := r.itemIndex[id]; ok && !b.excluded[ii] {
			b.excluded[ii] = true
			blackIdx = append(blackIdx, ii)
		}
	}
	// 返回前清理本次查询在复用暂存上留下的全部标记。
	defer func() {
		for _, e := range history {
			b.rated[e.itemIdx] = false
			b.histW[e.itemIdx] = 0
			b.excluded[e.itemIdx] = false
		}
		for _, ii := range blackIdx {
			b.excluded[ii] = false
		}
	}()

	// 第一遍：累计每个被触达候选的得分分子。
	// 候选 j 的分子 = sum_{i in 历史} w_u,i * sim(i,j)，
	// 其中 sim(i,j) = sum_v w_v,i * w_v,j / (|i| * |j|)。按评分者 v 聚合：
	// 先算 s_v = sum_{i in 历史∩items(v)} w_u,i * w_v,i / |i|，
	// 再对 v 的每个商品 j 累加 s_v * w_v,j / |j|，避免按用户对展开。
	// 权重为 0 或模长为 0（倒数模长为 0）的项对分子无贡献，直接跳过。
	var touched []int
	var sTouched []int
	if sumAbs > 0 {
		for _, hi := range history {
			if hi.weight == 0 {
				continue
			}
			invNi := r.itemInvN[hi.itemIdx]
			if invNi == 0 {
				continue
			}
			for _, ir := range r.byItem[hi.itemIdx] {
				if ir.weight == 0 {
					continue
				}
				v := ir.userIdx
				if !b.sSeen[v] {
					b.sSeen[v] = true
					sTouched = append(sTouched, v)
				}
				b.s[v] += hi.weight * (ir.weight * invNi)
			}
		}
		for _, v := range sTouched {
			sv := b.s[v]
			if sv == 0 {
				continue
			}
			for _, e := range r.byUser[v] {
				j := e.itemIdx
				if b.excluded[j] || e.weight == 0 {
					continue
				}
				nw := e.weight * r.itemInvN[j]
				if nw == 0 {
					continue
				}
				if !b.seen[j] {
					b.seen[j] = true
					touched = append(touched, j)
				}
				b.num[j] += sv * nw
			}
		}
	}
	defer func() {
		for _, v := range sTouched {
			b.s[v] = 0
			b.sSeen[v] = false
		}
		for _, j := range touched {
			b.num[j] = 0
			b.seen[j] = false
		}
	}()

	// 候选按得分分组：正分、零分（被触达但分子为 0）、负分。
	// 未被触达的候选得分也为 0，按商品ID升序与零分组合并。
	scoreOf := func(j int) float64 {
		return b.num[j] / (sumAbs * math.Log(2+float64(r.itemCount[j])))
	}
	sort.Slice(touched, func(a, b int) bool {
		sa, sb := scoreOf(touched[a]), scoreOf(touched[b])
		if sa != sb {
			return sa > sb
		}
		return r.itemIDs[touched[a]] < r.itemIDs[touched[b]]
	})
	var pos, zero, neg []int
	for _, j := range touched {
		s := scoreOf(j)
		switch {
		case s > 0:
			pos = append(pos, j)
		case s < 0:
			neg = append(neg, j)
		default:
			zero = append(zero, j)
		}
	}

	// 依次拼接：正分、零分（触达零分与未触达零分按商品ID升序合并）、负分。
	order := make([]int, 0, topN)
	order = append(order, pos...)
	zi := 0
	for _, j := range r.itemsByID {
		if b.excluded[j] {
			continue
		}
		isZero := zi < len(zero) && zero[zi] == j
		if b.seen[j] && !isZero {
			continue
		}
		if isZero {
			zi++
		}
		order = append(order, j)
	}
	order = append(order, neg...)
	if len(order) > topN {
		order = order[:topN]
	}

	// 第二遍：为最终结果计算贡献解释。
	recs := make([]Recommendation, 0, len(order))
	for _, j := range order {
		rec := Recommendation{ItemID: r.itemIDs[j]}
		if !b.seen[j] {
			// 零分候选：无贡献。
			recs = append(recs, rec)
			continue
		}
		rec.Score = scoreOf(j)
		rec.Top, rec.Other = r.contributions(j, sumAbs, rec.Score)
		recs = append(recs, rec)
	}
	return recs
}

// contributions 计算候选 j 的全部历史商品贡献，返回绝对贡献最大的
// 至多 3 个（绝对值相同按商品ID升序）及其余贡献之和。
// other = score - top 贡献之和，保证两部分相加精确还原 score。
func (r *Recommender) contributions(j int, sumAbs, score float64) ([]Contribution, float64) {
	b := &r.buf
	nj := r.itemNorm[j]
	denom := nj * sumAbs * math.Log(2+float64(r.itemCount[j]))
	// dot[i] = sum_v w_v,i * w_v,j，仅遍历同时评过 i 与 j 的用户。
	touched := b.dotTouched
	for _, ir := range r.byItem[j] {
		if ir.weight == 0 {
			continue
		}
		for _, e := range r.byUser[ir.userIdx] {
			if b.rated[e.itemIdx] && b.histW[e.itemIdx] != 0 {
				if !b.dotSeen[e.itemIdx] {
					b.dotSeen[e.itemIdx] = true
					touched = append(touched, e.itemIdx)
				}
				b.dot[e.itemIdx] += ir.weight * e.weight
			}
		}
	}
	contribs := make([]Contribution, 0, len(touched))
	for _, i := range touched {
		if ni := r.itemNorm[i]; ni != 0 {
			if v := b.histW[i] * b.dot[i] / (ni * denom); v != 0 {
				contribs = append(contribs, Contribution{ItemID: r.itemIDs[i], Value: v})
			}
		}
		// 清理暂存，供下一候选复用。
		b.dot[i] = 0
		b.dotSeen[i] = false
	}
	b.dotTouched = touched[:0]
	sort.Slice(contribs, func(a, b int) bool {
		aa, bb := math.Abs(contribs[a].Value), math.Abs(contribs[b].Value)
		if aa != bb {
			return aa > bb
		}
		return contribs[a].ItemID < contribs[b].ItemID
	})
	var top []Contribution
	topSum := 0.0
	for k, c := range contribs {
		if k < topContrib {
			top = append(top, c)
			topSum += c.Value
		}
	}
	return top, score - topSum
}
