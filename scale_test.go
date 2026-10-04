package itemcf

import (
	"math"
	"math/rand/v2"
	"runtime"
	"slices"
	"testing"
	"time"
)

const (
	scaleUsers   = 100000
	scaleItems   = 100000
	scaleRatings = 2000000
	scaleHeavyN  = 10    // heavy users ...
	scaleHeavyK  = 20000 // ... each with this many ratings
	scaleSuper   = 50000 // raters of the super-popular item
	scaleT       = 50000
	scaleLimit   = 60 * time.Second
	scaleMemCap  = 512 << 20
)

// genScaleRatings deterministically builds the required long-tail snapshot:
// 10 heavy users with 20000 ratings each, one item with 50000 raters,
// and random fill up to 2M unique (user,item) pairs.
func genScaleRatings() []Rating {
	rng := rand.New(rand.NewPCG(20261005, 98))
	seen := make(map[uint64]struct{}, scaleRatings)
	out := make([]Rating, 0, scaleRatings)
	add := func(u, it int) bool {
		key := uint64(u)<<32 | uint64(it)
		if _, dup := seen[key]; dup {
			return false
		}
		seen[key] = struct{}{}
		out = append(out, Rating{
			UserID: u, ItemID: it,
			Score: 1 + rng.IntN(5),
			Day:   scaleT - rng.IntN(366),
		})
		return true
	}
	// Heavy users share items [0,20000) so their histories fully overlap.
	for u := 0; u < scaleHeavyN; u++ {
		for it := 0; it < scaleHeavyK; it++ {
			add(u, it)
		}
	}
	// One super-popular item rated by 50000 distinct users.
	super := scaleItems / 2
	for u := scaleHeavyN; u < scaleHeavyN+scaleSuper; u++ {
		add(u, super)
	}
	for len(out) < scaleRatings {
		add(rng.IntN(scaleUsers), rng.IntN(scaleItems))
	}
	return out
}

// peakSampler records the maximum runtime MemStats Sys while running.
func peakSampler(stop <-chan struct{}) <-chan uint64 {
	peak := make(chan uint64, 1)
	go func() {
		var max uint64
		for {
			select {
			case <-stop:
				peak <- max
				return
			case <-time.After(2 * time.Millisecond):
				var m runtime.MemStats
				runtime.ReadMemStats(&m)
				if m.Sys > max {
					max = m.Sys
				}
			}
		}
	}()
	return peak
}

func TestScale(t *testing.T) {
	if testing.Short() {
		t.Skip("scale test skipped in -short mode")
	}
	stop := make(chan struct{})
	peakCh := peakSampler(stop)
	stopped := false
	defer func() {
		if !stopped {
			close(stop)
			<-peakCh
		}
	}()

	ratings := genScaleRatings()
	if len(ratings) != scaleRatings {
		t.Fatalf("generated %d ratings, want %d", len(ratings), scaleRatings)
	}

	start := time.Now()
	rec, err := Build(ratings, scaleT)
	if err != nil {
		t.Fatal(err)
	}
	buildDur := time.Since(start)

	// 100 query users: all 10 heavy users plus 90 distinct others.
	queryUsers := make([]int, 0, 100)
	for u := 0; u < scaleHeavyN; u++ {
		queryUsers = append(queryUsers, u)
	}
	rng := rand.New(rand.NewPCG(7, 7))
	seenQ := map[int]bool{}
	for len(queryUsers) < 100 {
		u := rng.IntN(scaleUsers)
		if !seenQ[u] {
			seenQ[u] = true
			queryUsers = append(queryUsers, u)
		}
	}
	blacklist := []int{0, 1, 2, scaleItems / 2, scaleItems - 1}

	start = time.Now()
	results := make([][]Recommendation, len(queryUsers))
	for i, u := range queryUsers {
		results[i] = rec.Recommend(u, blacklist)
		if len(results[i]) == 0 || len(results[i]) > 20 {
			t.Fatalf("user %d: got %d recommendations", u, len(results[i]))
		}
	}
	queryDur := time.Since(start)
	total := buildDur + queryDur

	close(stop)
	stopped = true
	peak := <-peakCh
	t.Logf("build=%v queries(100)=%v total=%v peakSys=%.1fMiB",
		buildDur, queryDur, total, float64(peak)/float64(1<<20))
	if total > scaleLimit {
		t.Fatalf("total CPU time %v exceeds %v", total, scaleLimit)
	}
	if peak > scaleMemCap {
		t.Fatalf("peak memory %d exceeds %d", peak, scaleMemCap)
	}

	// Independently recompute the FULL candidate ranking for a heavy and
	// a regular query user, and check the library's top-20 against it.
	for _, u := range []int{queryUsers[0], queryUsers[3], queryUsers[50], queryUsers[99]} {
		want := refFullRanking(ratings, u, blacklist)
		var got []Recommendation
		for i, qu := range queryUsers {
			if qu == u {
				got = results[i]
			}
		}
		compareRecs(t, got, want)
	}
}

// refFullRanking is an independent exact scorer: it iterates candidates
// one by one and accumulates co-occurrences per history item, then returns
// the top-20 of the complete candidate ranking.
func refFullRanking(ratings []Rating, queryUser int, blacklist []int) []Recommendation {
	// Per-user means.
	sum := map[int]float64{}
	cnt := map[int]int{}
	for _, rt := range ratings {
		sum[rt.UserID] += float64(rt.Score)
		cnt[rt.UserID]++
	}
	type ui struct {
		item int
		w    float64
	}
	userItems := map[int][]ui{}
	itemRaters := map[int][]ui{}
	norm := map[int]float64{}
	pop := map[int]int{}
	itemSet := map[int]bool{}
	for _, rt := range ratings {
		w := (float64(rt.Score) - sum[rt.UserID]/float64(cnt[rt.UserID])) *
			math.Pow(2, -float64(scaleT-rt.Day)/30)
		userItems[rt.UserID] = append(userItems[rt.UserID], ui{rt.ItemID, w})
		itemRaters[rt.ItemID] = append(itemRaters[rt.ItemID], ui{rt.UserID, w})
		norm[rt.ItemID] += w * w
		pop[rt.ItemID]++
		itemSet[rt.ItemID] = true
	}
	for it := range norm {
		norm[it] = math.Sqrt(norm[it])
	}
	qItems := userItems[queryUser]
	qW := make(map[int]float64, len(qItems))
	var sumAbs float64
	for _, x := range qItems {
		qW[x.item] = x.w
		sumAbs += math.Abs(x.w)
	}
	blocked := map[int]bool{}
	for _, id := range blacklist {
		blocked[id] = true
	}

	// Phase 1: s[v] = sum over history h of w_uh * w_vh / norm_h.
	s := map[int]float64{}
	if sumAbs != 0 {
		for _, h := range qItems {
			if norm[h.item] == 0 {
				continue
			}
			f := h.w / norm[h.item]
			for _, ra := range itemRaters[h.item] {
				s[ra.item] += f * ra.w
			}
		}
	}
	// Phase 2: raw[c] = sum over v of w_vc * s[v].
	raw := map[int]float64{}
	for v, sv := range s {
		if sv == 0 {
			continue
		}
		for _, x := range userItems[v] {
			raw[x.item] += x.w * sv
		}
	}
	// Phase 3: score every candidate and take the full ranking's top 20.
	var cands []Recommendation
	for c := range itemSet {
		if _, rated := qW[c]; rated || blocked[c] {
			continue
		}
		var score float64
		if sumAbs != 0 && norm[c] != 0 {
			score = raw[c] / (norm[c] * sumAbs * math.Log(2+float64(pop[c])))
		}
		cands = append(cands, Recommendation{ItemID: c, Score: score})
	}
	slices.SortFunc(cands, func(a, b Recommendation) int {
		if a.Score != b.Score {
			if a.Score > b.Score {
				return -1
			}
			return 1
		}
		return a.ItemID - b.ItemID
	})
	if len(cands) > 20 {
		cands = cands[:20]
	}
	// Phase 4: exact contributions for the winners, per candidate.
	for i := range cands {
		c := cands[i].ItemID
		cooc := map[int]float64{}
		for _, ra := range itemRaters[c] {
			for _, x := range userItems[ra.item] {
				if _, ok := qW[x.item]; ok {
					cooc[x.item] += x.w * ra.w
				}
			}
		}
		var contribs []Contribution
		for _, x := range qItems {
			var v float64
			if sumAbs != 0 && norm[c] != 0 && norm[x.item] != 0 {
				v = x.w * cooc[x.item] / (norm[x.item] * norm[c]) /
					sumAbs / math.Log(2+float64(pop[c]))
			}
			contribs = append(contribs, Contribution{ItemID: x.item, Value: v})
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
		cands[i].Top = slices.Clone(contribs[:nTop])
		var topSum float64
		for _, ct := range cands[i].Top {
			topSum += ct.Value
		}
		var rest float64
		for _, ct := range contribs[nTop:] {
			rest += ct.Value
		}
		cands[i].Rest = rest
		cands[i].Score = topSum + rest
	}
	return cands
}
