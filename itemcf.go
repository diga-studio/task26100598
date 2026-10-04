package itemcf

import (
	"math"
	"sort"
)

// userRating 是用户视角的一条加权评分。
type userRating struct {
	itemIdx int
	weight  float64 // 原始权重 w
}

// itemRating 是商品视角的一条加权评分（倒排索引项）。
type itemRating struct {
	userIdx int
	weight  float64 // 原始权重 w
}

// Recommender 保存一份固定评分快照的推荐数据。
type Recommender struct {
	userIDs   []int          // 用户内部下标 -> 用户ID
	itemIDs   []int          // 商品内部下标 -> 商品ID
	userIndex map[int]int    // 用户ID -> 内部下标
	itemIndex map[int]int    // 商品ID -> 内部下标
	byUser    [][]userRating // 用户 -> 其评分（含权重）
	byItem    [][]itemRating // 商品 -> 评分它的用户（含权重）
	itemNorm  []float64      // 商品权重向量的模长
	itemInvN  []float64      // 商品模长的倒数（模长为 0 时为 0）
	itemCount []int          // 商品的评分人数（含权重为 0 的评分）
	itemsByID []int          // 按商品ID升序排列的商品内部下标
	buf       queryBuffers   // 跨查询复用的暂存（不支持并发查询）
}

// queryBuffers 是单次推荐的暂存空间，复用以避免每次查询重复分配。
type queryBuffers struct {
	num        []float64 // 候选得分分子
	histW      []float64 // 用户历史权重查表
	s          []float64 // 评分者聚合权重
	dot        []float64 // 贡献点积
	seen       []bool    // 候选已触达标记
	rated      []bool    // 用户已评标记
	excluded   []bool    // 候选排除标记
	sSeen      []bool    // 评分者已触达标记
	dotSeen    []bool    // 贡献已触达标记
	dotTouched []int
}

// lazyInit 按快照规模惰性分配暂存。
func (b *queryBuffers) lazyInit(numUsers, numItems int) {
	if len(b.num) == numItems {
		return
	}
	b.num = make([]float64, numItems)
	b.histW = make([]float64, numItems)
	b.s = make([]float64, numUsers)
	b.dot = make([]float64, numItems)
	b.seen = make([]bool, numItems)
	b.rated = make([]bool, numItems)
	b.excluded = make([]bool, numItems)
	b.sSeen = make([]bool, numUsers)
	b.dotSeen = make([]bool, numItems)
}

// Build 从完整评分快照构建推荐数据，refDay 为参考日 T。
// 每条记录权重 = (评分 - 用户均分) * 2^(-(T-评分日)/30)。
func Build(ratings []Rating, refDay int) *Recommender {
	r := &Recommender{
		userIndex: make(map[int]int, len(ratings)/2),
		itemIndex: make(map[int]int, len(ratings)/2),
	}
	// 第一遍：建立用户、商品下标，累计用户评分和与双方评分条数。
	var userScoreSum []int64
	var userScoreCnt []int64
	for _, rt := range ratings {
		ui, ok := r.userIndex[rt.UserID]
		if !ok {
			ui = len(r.userIDs)
			r.userIndex[rt.UserID] = ui
			r.userIDs = append(r.userIDs, rt.UserID)
			r.byUser = append(r.byUser, nil)
			userScoreSum = append(userScoreSum, 0)
			userScoreCnt = append(userScoreCnt, 0)
		}
		ii, ok := r.itemIndex[rt.ItemID]
		if !ok {
			ii = len(r.itemIDs)
			r.itemIndex[rt.ItemID] = ii
			r.itemIDs = append(r.itemIDs, rt.ItemID)
			r.byItem = append(r.byItem, nil)
			r.itemNorm = append(r.itemNorm, 0)
			r.itemInvN = append(r.itemInvN, 0)
			r.itemCount = append(r.itemCount, 0)
		}
		userScoreSum[ui] += int64(rt.Score)
		userScoreCnt[ui]++
		r.itemCount[ii]++
	}
	// 用户均分 = 全部评分的算术平均。
	userMean := make([]float64, len(r.userIDs))
	for u := range userMean {
		userMean[u] = float64(userScoreSum[u]) / float64(userScoreCnt[u])
	}
	// 按精确条数预分配索引空间，避免 append 增长产生内存垃圾。
	for u := range r.byUser {
		r.byUser[u] = make([]userRating, 0, userScoreCnt[u])
	}
	for i := range r.byItem {
		r.byItem[i] = make([]itemRating, 0, r.itemCount[i])
	}
	// 第二遍：计算权重，填充双向索引与商品模长、评分人数。
	normSq := make([]float64, len(r.itemIDs))
	for _, rt := range ratings {
		ui, ii := r.userIndex[rt.UserID], r.itemIndex[rt.ItemID]
		w := (float64(rt.Score) - userMean[ui]) * math.Pow(2, -float64(refDay-rt.Day)/30)
		r.byUser[ui] = append(r.byUser[ui], userRating{itemIdx: ii, weight: w})
		r.byItem[ii] = append(r.byItem[ii], itemRating{userIdx: ui, weight: w})
		normSq[ii] += w * w
	}
	for ii := range r.itemNorm {
		r.itemNorm[ii] = math.Sqrt(normSq[ii])
		if r.itemNorm[ii] != 0 {
			r.itemInvN[ii] = 1 / r.itemNorm[ii]
		}
	}
	r.itemsByID = make([]int, len(r.itemIDs))
	for ii := range r.itemsByID {
		r.itemsByID[ii] = ii
	}
	sort.Slice(r.itemsByID, func(a, b int) bool {
		return r.itemIDs[r.itemsByID[a]] < r.itemIDs[r.itemsByID[b]]
	})
	return r
}
