package itemcf

import (
	"math"
	"slices"
	"testing"
)

// --- Independent reference implementation, straight from the definitions ---

type refModel struct {
	T      int
	users  []int
	items  []int
	weight map[[2]int]float64 // (user,item) -> weight
	pop    map[int]int
	norm   map[int]float64
	hist   map[int][]int // user -> rated items
	sumAbs map[int]float64
}

func buildRef(ratings []Rating, T int) *refModel {
	m := &refModel{
		T: T, weight: map[[2]int]float64{}, pop: map[int]int{},
		norm: map[int]float64{}, hist: map[int][]int{}, sumAbs: map[int]float64{},
	}
	sum := map[int]float64{}
	cnt := map[int]int{}
	seenU, seenI := map[int]bool{}, map[int]bool{}
	for _, rt := range ratings {
		sum[rt.UserID] += float64(rt.Score)
		cnt[rt.UserID]++
		if !seenU[rt.UserID] {
			seenU[rt.UserID] = true
			m.users = append(m.users, rt.UserID)
		}
		if !seenI[rt.ItemID] {
			seenI[rt.ItemID] = true
			m.items = append(m.items, rt.ItemID)
		}
	}
	slices.Sort(m.items)
	for _, rt := range ratings {
		mean := sum[rt.UserID] / float64(cnt[rt.UserID])
		w := (float64(rt.Score) - mean) * math.Pow(2, -float64(T-rt.Day)/30)
		m.weight[[2]int{rt.UserID, rt.ItemID}] = w
		m.pop[rt.ItemID]++
		m.norm[rt.ItemID] += w * w
		m.hist[rt.UserID] = append(m.hist[rt.UserID], rt.ItemID)
		m.sumAbs[rt.UserID] += math.Abs(w)
	}
	for i := range m.norm {
		m.norm[i] = math.Sqrt(m.norm[i])
	}
	return m
}

func (m *refModel) sim(a, b int) float64 {
	if m.norm[a] == 0 || m.norm[b] == 0 {
		return 0
	}
	var cooc float64
	for _, u := range m.users {
		wa, oka := m.weight[[2]int{u, a}]
		wb, okb := m.weight[[2]int{u, b}]
		if oka && okb {
			cooc += wa * wb
		}
	}
	return cooc / (m.norm[a] * m.norm[b])
}

func (m *refModel) contrib(u, h, c int) float64 {
	if m.sumAbs[u] == 0 {
		return 0
	}
	return m.weight[[2]int{u, h}] * m.sim(h, c) /
		(m.sumAbs[u] * math.Log(2+float64(m.pop[c])))
}

func (m *refModel) recommend(u int, blacklist []int) []Recommendation {
	blocked := map[int]bool{}
	for _, id := range blacklist {
		blocked[id] = true
	}
	inHist := map[int]bool{}
	for _, h := range m.hist[u] {
		inHist[h] = true
	}
	var recs []Recommendation
	for _, c := range m.items {
		if inHist[c] || blocked[c] {
			continue
		}
		var contribs []Contribution
		for _, h := range m.hist[u] {
			contribs = append(contribs, Contribution{ItemID: h, Value: m.contrib(u, h, c)})
		}
		slices.SortFunc(contribs, func(a, b Contribution) int {
			aa, bb := math.Abs(a.Value), math.Abs(b.Value)
			if aa != bb {
				if aa > bb {
					return -1
				}
				return 1
			}
			return a.ItemID - b.ItemID
		})
		nTop := 3
		if len(contribs) < nTop {
			nTop = len(contribs)
		}
		rec := Recommendation{ItemID: c, Top: slices.Clone(contribs[:nTop])}
		var topSum float64
		for _, ct := range rec.Top {
			topSum += ct.Value
		}
		var rest float64
		for _, ct := range contribs[nTop:] {
			rest += ct.Value
		}
		rec.Rest = rest
		rec.Score = topSum + rest
		recs = append(recs, rec)
	}
	slices.SortFunc(recs, func(a, b Recommendation) int {
		if a.Score != b.Score {
			if a.Score > b.Score {
				return -1
			}
			return 1
		}
		return a.ItemID - b.ItemID
	})
	if len(recs) > 20 {
		recs = recs[:20]
	}
	return recs
}

// --- tests ---

func almostEq(a, b float64) bool { return math.Abs(a-b) <= 1e-9 }

func compareRecs(t *testing.T, got, want []Recommendation) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("result count: got %d, want %d", len(got), len(want))
	}
	for i := range got {
		g, w := got[i], want[i]
		if g.ItemID != w.ItemID {
			t.Fatalf("position %d: got item %d, want item %d", i, g.ItemID, w.ItemID)
		}
		if !almostEq(g.Score, w.Score) {
			t.Fatalf("item %d: score got %v, want %v", g.ItemID, g.Score, w.Score)
		}
		if len(g.Top) != len(w.Top) {
			t.Fatalf("item %d: top count got %d, want %d", g.ItemID, len(g.Top), len(w.Top))
		}
		for j := range g.Top {
			if g.Top[j].ItemID != w.Top[j].ItemID || !almostEq(g.Top[j].Value, w.Top[j].Value) {
				t.Fatalf("item %d top %d: got %+v, want %+v", g.ItemID, j, g.Top[j], w.Top[j])
			}
		}
		if !almostEq(g.Rest, w.Rest) {
			t.Fatalf("item %d: rest got %v, want %v", g.ItemID, g.Rest, w.Rest)
		}
		var topSum float64
		for _, ct := range g.Top {
			topSum += ct.Value
		}
		if topSum+g.Rest != g.Score {
			t.Fatalf("item %d: top+rest %v does not restore score %v", g.ItemID, topSum+g.Rest, g.Score)
		}
	}
}

// richSnapshot has varied days, an exact-mean (zero-weight) rating in
// user 1's history, and a single-rating user 7 (all weights zero).
func richSnapshot() []Rating {
	return []Rating{
		{UserID: 1, ItemID: 10, Score: 5, Day: 1000},
		{UserID: 1, ItemID: 11, Score: 3, Day: 970}, // user1 mean=3 -> zero weight
		{UserID: 1, ItemID: 12, Score: 1, Day: 640},
		{UserID: 2, ItemID: 10, Score: 5, Day: 999},
		{UserID: 2, ItemID: 13, Score: 5, Day: 1000},
		{UserID: 2, ItemID: 14, Score: 1, Day: 800},
		{UserID: 3, ItemID: 11, Score: 2, Day: 1000},
		{UserID: 3, ItemID: 13, Score: 4, Day: 990},
		{UserID: 3, ItemID: 15, Score: 3, Day: 635},
		{UserID: 4, ItemID: 10, Score: 4, Day: 1000},
		{UserID: 4, ItemID: 16, Score: 2, Day: 1000},
		{UserID: 5, ItemID: 12, Score: 5, Day: 700},
		{UserID: 5, ItemID: 14, Score: 4, Day: 1000},
		{UserID: 5, ItemID: 17, Score: 3, Day: 1000},
		{UserID: 6, ItemID: 15, Score: 1, Day: 1000},
		{UserID: 6, ItemID: 16, Score: 5, Day: 900},
		{UserID: 6, ItemID: 18, Score: 3, Day: 1000},
		{UserID: 7, ItemID: 19, Score: 4, Day: 1000}, // single rating: weight 0
		{UserID: 8, ItemID: 13, Score: 2, Day: 1000},
		{UserID: 8, ItemID: 17, Score: 5, Day: 1000},
		{UserID: 8, ItemID: 18, Score: 1, Day: 1000},
	}
}

func TestMatchesReference(t *testing.T) {
	ratings := richSnapshot()
	const T = 1000
	rec, err := Build(ratings, T)
	if err != nil {
		t.Fatal(err)
	}
	ref := buildRef(ratings, T)
	for _, u := range []int{1, 2, 3, 4, 5, 6, 7, 8} {
		for _, bl := range [][]int{nil, {13}, {13, 16, 999}} {
			got := rec.Recommend(u, bl)
			want := ref.recommend(u, bl)
			compareRecs(t, got, want)
		}
	}
}

func TestUnknownUser(t *testing.T) {
	rec, err := Build(richSnapshot(), 1000)
	if err != nil {
		t.Fatal(err)
	}
	if got := rec.Recommend(424242, nil); got != nil {
		t.Fatalf("unknown user: got %v, want nil", got)
	}
}

func TestValidation(t *testing.T) {
	good := []Rating{{UserID: 1, ItemID: 1, Score: 3, Day: 100}}
	if _, err := Build(good, 100); err != nil {
		t.Fatalf("valid snapshot rejected: %v", err)
	}
	cases := map[string][]Rating{
		"score low":  {{UserID: 1, ItemID: 1, Score: 0, Day: 100}},
		"score high": {{UserID: 1, ItemID: 1, Score: 6, Day: 100}},
		"day future": {{UserID: 1, ItemID: 1, Score: 3, Day: 101}},
		"day old":    {{UserID: 1, ItemID: 1, Score: 3, Day: -266}}, // T-366
		"duplicate": {
			{UserID: 1, ItemID: 1, Score: 3, Day: 100},
			{UserID: 1, ItemID: 1, Score: 4, Day: 99},
		},
	}
	for name, rs := range cases {
		if _, err := Build(rs, 100); err == nil {
			t.Fatalf("%s: expected error, got nil", name)
		}
	}
	if _, err := Build([]Rating{{UserID: 1, ItemID: 1, Score: 3, Day: -265}}, 100); err != nil {
		t.Fatalf("boundary day T-365 rejected: %v", err)
	}
}
