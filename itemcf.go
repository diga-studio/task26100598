// Package itemcf implements an ItemCF recommender over a fixed rating
// snapshot. All data lives in process memory.
package itemcf

// Rating is a single user-item rating record.
type Rating struct {
	UserID int
	ItemID int
	Score  int // 1..5
	Day    int // rating day, must be within [RefDay-365, RefDay]
}

// Contribution is one historical item's signed contribution to a
// recommendation's final score.
type Contribution struct {
	ItemID int
	Value  float64
}

// Recommendation is one recommended item with its score and explanation.
// The sum of Top contributions plus Rest equals Score.
type Recommendation struct {
	ItemID int
	Score  float64
	Top    []Contribution // up to 3, sorted by |value| desc, ItemID asc
	Rest   float64        // sum of all remaining contributions
}
