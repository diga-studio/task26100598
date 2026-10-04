package itemcf

import "testing"

// blacklistSnapshot: item 12 scores positive for user 1; item 14 has no
// common raters with user 1's history (zero score); item 13 scores negative.
func blacklistSnapshot() []Rating {
	return []Rating{
		{UserID: 1, ItemID: 10, Score: 5, Day: 100},
		{UserID: 1, ItemID: 11, Score: 1, Day: 100},
		{UserID: 2, ItemID: 10, Score: 5, Day: 100},
		{UserID: 2, ItemID: 12, Score: 5, Day: 100},
		{UserID: 2, ItemID: 13, Score: 1, Day: 100},
		{UserID: 3, ItemID: 11, Score: 2, Day: 100},
		{UserID: 3, ItemID: 12, Score: 4, Day: 100},
		{UserID: 4, ItemID: 14, Score: 3, Day: 100},
	}
}

func TestBlacklistKeepsZeroAboveNegative(t *testing.T) {
	rec, err := Build(blacklistSnapshot(), 100)
	if err != nil {
		t.Fatal(err)
	}
	// Sanity: without blacklist, item 12 is the positive top pick.
	top := rec.Recommend(1, nil)
	if len(top) == 0 || top[0].ItemID != 12 || top[0].Score <= 0 {
		t.Fatalf("expected item 12 with positive score first, got %+v", top)
	}
	// Blacklist the popular positive item: zero-score 14 must outrank
	// negative-score 13, and 12 must be gone.
	got := rec.Recommend(1, []int{12})
	var i13, i14 = -1, -1
	for i, r := range got {
		if r.ItemID == 12 {
			t.Fatalf("blacklisted item 12 present: %+v", got)
		}
		if r.ItemID == 13 {
			i13 = i
			if r.Score >= 0 {
				t.Fatalf("item 13 score = %v, want negative", r.Score)
			}
		}
		if r.ItemID == 14 {
			i14 = i
			if r.Score != 0 {
				t.Fatalf("item 14 score = %v, want exactly 0", r.Score)
			}
		}
	}
	if i14 < 0 || i13 < 0 || i14 > i13 {
		t.Fatalf("zero-score item must precede negative-score item: %+v", got)
	}
}

// TestMeanCancelsAllWeights: user 1's ratings are all equal, so every
// weight is zero and every candidate scores exactly 0, ordered by item ID.
func TestMeanCancelsAllWeights(t *testing.T) {
	ratings := []Rating{
		{UserID: 1, ItemID: 10, Score: 3, Day: 100},
		{UserID: 1, ItemID: 11, Score: 3, Day: 90},
		{UserID: 1, ItemID: 12, Score: 3, Day: 80},
		{UserID: 2, ItemID: 13, Score: 5, Day: 100},
		{UserID: 2, ItemID: 14, Score: 1, Day: 100},
	}
	rec, err := Build(ratings, 100)
	if err != nil {
		t.Fatal(err)
	}
	got := rec.Recommend(1, nil)
	wantIDs := []int{13, 14}
	if len(got) != len(wantIDs) {
		t.Fatalf("got %+v, want items %v", got, wantIDs)
	}
	for i, r := range got {
		if r.ItemID != wantIDs[i] || r.Score != 0 {
			t.Fatalf("position %d: got (%d,%v), want (%d,0)", i, r.ItemID, r.Score, wantIDs[i])
		}
		if len(r.Top) != 3 { // 3 history items, all zero contribution
			t.Fatalf("top count = %d, want 3", len(r.Top))
		}
		for _, ct := range r.Top {
			if ct.Value != 0 {
				t.Fatalf("contribution = %v, want 0", ct.Value)
			}
		}
		if r.Rest != 0 {
			t.Fatalf("rest = %v, want 0", r.Rest)
		}
	}
	// Top ties on |value| break by item ID ascending.
	if got[0].Top[0].ItemID != 10 || got[0].Top[1].ItemID != 11 || got[0].Top[2].ItemID != 12 {
		t.Fatalf("zero-contribution tie order wrong: %+v", got[0].Top)
	}
}
