package itemcf

import (
	"fmt"
	"math"
	"slices"
)

// uItem is one weighted item in a user's history.
type uItem struct {
	item int32
	w    float64
}

// iRater is one weighted user in an item's rater list.
type iRater struct {
	user int32
	w    float64
}

// Recommender holds the built model for one fixed rating snapshot.
type Recommender struct {
	refDay int

	userIDs  []int         // user index -> original user ID
	itemIDs  []int         // item index -> original item ID
	userIdx  map[int]int32 // original user ID -> user index
	itemIdx  map[int]int32 // original item ID -> item index
	userOff  []int64       // CSR offsets into uItems, len nUsers+1
	uItems   []uItem       // per-user history, sorted by item index
	sumAbs   []float64     // per-user sum of |weight|
	itemOff  []int64       // CSR offsets into iRaters, len nItems+1
	iRaters  []iRater      // per-item raters
	itemNorm []float64     // per-item L2 norm of weights over all users
	itemPop  []int32       // per-item rating count (includes zero-weight ratings)

	entries []entry // transient, released after build
}

// entry is a transient record used while building.
type entry struct {
	user  int32
	item  int32
	score int32
	day   int32
}

// Build validates the snapshot and constructs a Recommender.
// refDay is the reference day T; every rating day must be in [T-365, T].
func Build(ratings []Rating, refDay int) (*Recommender, error) {
	r := &Recommender{refDay: refDay}
	if err := r.index(ratings); err != nil {
		return nil, err
	}
	r.weigh()
	return r, nil
}

// index validates records, assigns dense user/item indices, and sorts
// entries by (user, item) so duplicates are adjacent.
func (r *Recommender) index(ratings []Rating) error {
	r.userIdx = make(map[int]int32)
	r.itemIdx = make(map[int]int32)
	entries := make([]entry, len(ratings))
	minDay := r.refDay - 365
	for i, rt := range ratings {
		if rt.Score < 1 || rt.Score > 5 {
			return fmt.Errorf("itemcf: rating %d: score %d out of range [1,5]", i, rt.Score)
		}
		if rt.Day < minDay || rt.Day > r.refDay {
			return fmt.Errorf("itemcf: rating %d: day %d out of range [%d,%d]", i, rt.Day, minDay, r.refDay)
		}
		u, ok := r.userIdx[rt.UserID]
		if !ok {
			u = int32(len(r.userIDs))
			r.userIdx[rt.UserID] = u
			r.userIDs = append(r.userIDs, rt.UserID)
		}
		it, ok := r.itemIdx[rt.ItemID]
		if !ok {
			it = int32(len(r.itemIDs))
			r.itemIdx[rt.ItemID] = it
			r.itemIDs = append(r.itemIDs, rt.ItemID)
		}
		entries[i] = entry{user: u, item: it, score: int32(rt.Score), day: int32(rt.Day)}
	}
	slices.SortFunc(entries, func(a, b entry) int {
		if a.user != b.user {
			return int(a.user) - int(b.user)
		}
		return int(a.item) - int(b.item)
	})
	for i := 1; i < len(entries); i++ {
		if entries[i].user == entries[i-1].user && entries[i].item == entries[i-1].item {
			return fmt.Errorf("itemcf: duplicate rating for user %d item %d",
				r.userIDs[entries[i].user], r.itemIDs[entries[i].item])
		}
	}
	r.entries = entries
	return nil
}

// weigh computes user means, per-record weights, and both CSR indexes.
func (r *Recommender) weigh() {
	nUsers := len(r.userIDs)
	nItems := len(r.itemIDs)
	entries := r.entries

	r.userOff = make([]int64, nUsers+1)
	r.itemPop = make([]int32, nItems)
	for _, e := range entries {
		r.userOff[e.user+1]++
		r.itemPop[e.item]++
	}
	for i := 0; i < nUsers; i++ {
		r.userOff[i+1] += r.userOff[i]
	}
	r.itemOff = make([]int64, nItems+1)
	for i := 0; i < nItems; i++ {
		r.itemOff[i+1] = r.itemOff[i] + int64(r.itemPop[i])
	}

	// User means over all their ratings.
	means := make([]float64, nUsers)
	for u := 0; u < nUsers; u++ {
		var sum float64
		for k := r.userOff[u]; k < r.userOff[u+1]; k++ {
			sum += float64(entries[k].score)
		}
		means[u] = sum / float64(r.userOff[u+1]-r.userOff[u])
	}

	r.uItems = make([]uItem, len(entries))
	r.sumAbs = make([]float64, nUsers)
	r.iRaters = make([]iRater, len(entries))
	r.itemNorm = make([]float64, nItems)
	cursor := make([]int64, nItems)
	copy(cursor, r.itemOff[:nItems])
	for u := 0; u < nUsers; u++ {
		mean := means[u]
		var sa float64
		for k := r.userOff[u]; k < r.userOff[u+1]; k++ {
			e := entries[k]
			w := (float64(e.score) - mean) * math.Pow(2, -float64(r.refDay-int(e.day))/30)
			r.uItems[k] = uItem{item: e.item, w: w}
			sa += math.Abs(w)
			p := cursor[e.item]
			r.iRaters[p] = iRater{user: int32(u), w: w}
			cursor[e.item]++
			r.itemNorm[e.item] += w * w
		}
		r.sumAbs[u] = sa
	}
	for i := range r.itemNorm {
		r.itemNorm[i] = math.Sqrt(r.itemNorm[i])
	}
	r.entries = nil
}
