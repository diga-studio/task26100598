package itemcf

import (
	"math"
	"sort"
	"testing"
)

// ---------- 独立参考实现：直接按定义暴力计算 ----------

type refModel struct {
	wu    map[int]map[int]float64 // 用户 -> 商品 -> 权重
	wi    map[int]map[int]float64 // 商品 -> 用户 -> 权重
	norm  map[int]float64
	count map[int]int
	items []int // 升序
}

func buildRef(ratings []Rating, refDay int) *refModel {
	sum := map[int]int{}
	cnt := map[int]int{}
	for _, rt := range ratings {
		sum[rt.UserID] += rt.Score
		cnt[rt.UserID]++
	}
	m := &refModel{
		wu:    map[int]map[int]float64{},
		wi:    map[int]map[int]float64{},
		norm:  map[int]float64{},
		count: map[int]int{},
	}
	itemSet := map[int]bool{}
	for _, rt := range ratings {
		mean := float64(sum[rt.UserID]) / float64(cnt[rt.UserID])
		w := (float64(rt.Score) - mean) * math.Pow(2, -float64(refDay-rt.Day)/30)
		if m.wu[rt.UserID] == nil {
			m.wu[rt.UserID] = map[int]float64{}
		}
		if m.wi[rt.ItemID] == nil {
			m.wi[rt.ItemID] = map[int]float64{}
		}
		m.wu[rt.UserID][rt.ItemID] = w
		m.wi[rt.ItemID][rt.UserID] = w
		m.norm[rt.ItemID] += w * w
		m.count[rt.ItemID]++
		itemSet[rt.ItemID] = true
	}
	for it := range m.norm {
		m.norm[it] = math.Sqrt(m.norm[it])
	}
	for it := range itemSet {
		m.items = append(m.items, it)
	}
	sort.Ints(m.items)
	return m
}

// sim 按定义计算两个商品在全部用户坐标上的权重余弦。
func (m *refModel) sim(a, b int) float64 {
	na, nb := m.norm[a], m.norm[b]
	if na == 0 || nb == 0 {
		return 0
	}
	ua, ub := m.wi[a], m.wi[b]
	if len(ua) > len(ub) {
		ua, ub = ub, ua
	}
	dot := 0.0
	for u, wa := range ua {
		if wb, ok := ub[u]; ok {
			dot += wa * wb
		}
	}
	return dot / (na * nb)
}

type refRec struct {
	item  int
	score float64
	top   []Contribution
	other float64
}

func (m *refModel) recommend(userID int, blacklist []int) []refRec {
	hist := m.wu[userID]
	excluded := map[int]bool{}
	for it := range hist {
		excluded[it] = true
	}
	for _, id := range blacklist {
		excluded[id] = true
	}
	sumAbs := 0.0
	for _, w := range hist {
		sumAbs += math.Abs(w)
	}
	var cands []refRec
	for _, it := range m.items {
		if excluded[it] {
			continue
		}
		denom := sumAbs * math.Log(2+float64(m.count[it]))
		score := 0.0
		var contribs []Contribution
		if sumAbs > 0 {
			for hi, wu := range hist {
				c := wu * m.sim(hi, it) / denom
				score += c
				if c != 0 {
					contribs = append(contribs, Contribution{ItemID: hi, Value: c})
				}
			}
		}
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
			if k < 3 {
				top = append(top, c)
				topSum += c.Value
			}
		}
		cands = append(cands, refRec{item: it, score: score, top: top, other: score - topSum})
	}
	sort.Slice(cands, func(a, b int) bool {
		if cands[a].score != cands[b].score {
			return cands[a].score > cands[b].score
		}
		return cands[a].item < cands[b].item
	})
	if len(cands) > 20 {
		cands = cands[:20]
	}
	return cands
}

// compareRecs 比对库输出与参考输出：排序一致，数值绝对误差 <= 1e-9。
func compareRecs(t *testing.T, got []Recommendation, want []refRec) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("结果条数不同: got %d, want %d", len(got), len(want))
	}
	for k := range got {
		g, w := got[k], want[k]
		if g.ItemID != w.item {
			t.Fatalf("第 %d 名商品不同: got %d, want %d", k, g.ItemID, w.item)
		}
		if math.Abs(g.Score-w.score) > 1e-9 {
			t.Fatalf("商品 %d 得分误差过大: got %v, want %v", g.ItemID, g.Score, w.score)
		}
		if len(g.Top) != len(w.top) {
			t.Fatalf("商品 %d 贡献条数不同: got %d, want %d", g.ItemID, len(g.Top), len(w.top))
		}
		for c := range g.Top {
			if g.Top[c].ItemID != w.top[c].ItemID {
				t.Fatalf("商品 %d 第 %d 个贡献商品不同: got %d, want %d",
					g.ItemID, c, g.Top[c].ItemID, w.top[c].ItemID)
			}
			if math.Abs(g.Top[c].Value-w.top[c].Value) > 1e-9 {
				t.Fatalf("商品 %d 贡献 %d 数值误差过大: got %v, want %v",
					g.ItemID, g.Top[c].ItemID, g.Top[c].Value, w.top[c].Value)
			}
		}
		if math.Abs(g.Other-w.other) > 1e-9 {
			t.Fatalf("商品 %d 其余贡献和误差过大: got %v, want %v", g.ItemID, g.Other, w.other)
		}
		// 两部分相加必须精确还原得分。
		topSum := 0.0
		for _, c := range g.Top {
			topSum += c.Value
		}
		if topSum+g.Other != g.Score {
			t.Fatalf("商品 %d 贡献无法还原得分: %v + %v != %v", g.ItemID, topSum, g.Other, g.Score)
		}
	}
}

// ---------- 测试 ----------

const testRefDay = 10000

// smallRatings 覆盖：不同评分日、权重为 0 的评分（评分等于用户均分）、
// 模长为 0 的商品、正分/零分/负分候选。
func smallRatings() []Rating {
	return []Rating{
		// 用户 1（目标用户）：均分 3
		{UserID: 1, ItemID: 101, Score: 5, Day: 10000},
		{UserID: 1, ItemID: 102, Score: 1, Day: 9970},
		{UserID: 1, ItemID: 103, Score: 3, Day: 9900}, // 权重为 0
		// 用户 2：均分 10/3
		{UserID: 2, ItemID: 101, Score: 4, Day: 10000},
		{UserID: 2, ItemID: 104, Score: 5, Day: 9940},
		{UserID: 2, ItemID: 105, Score: 1, Day: 10000},
		// 用户 3：均分 2
		{UserID: 3, ItemID: 102, Score: 1, Day: 10000},
		{UserID: 3, ItemID: 104, Score: 2, Day: 9850},
		{UserID: 3, ItemID: 106, Score: 3, Day: 10000},
		// 用户 4：只评一件，其商品模长为 0
		{UserID: 4, ItemID: 107, Score: 4, Day: 10000},
		// 用户 5：均分 4
		{UserID: 5, ItemID: 105, Score: 5, Day: 10000},
		{UserID: 5, ItemID: 106, Score: 3, Day: 9700},
		{UserID: 5, ItemID: 108, Score: 4, Day: 10000}, // 权重为 0
	}
}

func TestSmallDatasetMatchesReference(t *testing.T) {
	ratings := smallRatings()
	rec := Build(ratings, testRefDay)
	ref := buildRef(ratings, testRefDay)
	for _, userID := range []int{1, 2, 3, 4, 5} {
		for _, blacklist := range [][]int{nil, {104}, {101, 106, 999}} {
			got := rec.Recommend(userID, blacklist)
			want := ref.recommend(userID, blacklist)
			compareRecs(t, got, want)
		}
	}
}

// TestBlacklistedPositiveZeroBeforeNegative 热门正分商品被屏蔽后，
// 零分候选仍排在负分候选前面。
func TestBlacklistedPositiveZeroBeforeNegative(t *testing.T) {
	ratings := []Rating{
		{UserID: 1, ItemID: 10, Score: 5, Day: 10000}, // 用户1均分3：w(10)=2, w(11)=-2
		{UserID: 1, ItemID: 11, Score: 1, Day: 10000},
		{UserID: 2, ItemID: 10, Score: 4, Day: 10000}, // 用户2均分10/3
		{UserID: 2, ItemID: 12, Score: 5, Day: 10000},
		{UserID: 2, ItemID: 13, Score: 1, Day: 10000},
		{UserID: 3, ItemID: 14, Score: 3, Day: 10000}, // 用户3只评一件，商品14模长为0
	}
	rec := Build(ratings, testRefDay)

	// 不屏蔽时：商品 12 正分，商品 13 负分，商品 14 零分。
	got := rec.Recommend(1, nil)
	if len(got) != 3 || got[0].ItemID != 12 || got[1].ItemID != 14 || got[2].ItemID != 13 {
		t.Fatalf("未屏蔽时排名不符预期: %+v", got)
	}
	if got[0].Score <= 0 || got[1].Score != 0 || got[2].Score >= 0 {
		t.Fatalf("未屏蔽时得分符号不符预期: %+v", got)
	}

	// 屏蔽热门正分商品 12 后：零分的 14 必须排在负分的 13 前面。
	got = rec.Recommend(1, []int{12})
	if len(got) != 2 || got[0].ItemID != 14 || got[1].ItemID != 13 {
		t.Fatalf("屏蔽后排名不符预期: %+v", got)
	}
	if got[0].Score != 0 || got[1].Score >= 0 {
		t.Fatalf("屏蔽后得分符号不符预期: %+v", got)
	}
}

// TestMeanCancelsAllWeights 用户均分消去全部权重时，所有候选得分为 0，
// 按商品ID升序返回。
func TestMeanCancelsAllWeights(t *testing.T) {
	ratings := []Rating{
		{UserID: 1, ItemID: 5, Score: 3, Day: 10000},
		{UserID: 1, ItemID: 6, Score: 3, Day: 9900},
		{UserID: 2, ItemID: 5, Score: 4, Day: 10000},
		{UserID: 2, ItemID: 7, Score: 2, Day: 10000},
		{UserID: 3, ItemID: 8, Score: 5, Day: 10000},
		{UserID: 3, ItemID: 9, Score: 1, Day: 10000},
	}
	rec := Build(ratings, testRefDay)
	got := rec.Recommend(1, nil)
	// 候选为 7、8、9，全部零分，按商品ID升序。
	if len(got) != 3 || got[0].ItemID != 7 || got[1].ItemID != 8 || got[2].ItemID != 9 {
		t.Fatalf("排名不符预期: %+v", got)
	}
	for _, g := range got {
		if g.Score != 0 || len(g.Top) != 0 || g.Other != 0 {
			t.Fatalf("零分结果不符预期: %+v", g)
		}
	}
	// 参考实现交叉验证。
	ref := buildRef(ratings, testRefDay)
	compareRecs(t, got, ref.recommend(1, nil))
}

func TestUnknownUser(t *testing.T) {
	rec := Build(smallRatings(), testRefDay)
	if got := rec.Recommend(999, nil); got != nil {
		t.Fatalf("未知用户应返回 nil, got %+v", got)
	}
}
