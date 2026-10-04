package itemcf

// Rating 是一条评分记录。用户与商品的组合不重复。
type Rating struct {
	UserID int
	ItemID int
	Score  int // 1 至 5
	Day    int // 整数评分日
}

// Contribution 是某个历史商品对推荐得分的带符号贡献。
type Contribution struct {
	ItemID int
	Value  float64
}

// Recommendation 是一条推荐结果。
type Recommendation struct {
	ItemID int
	Score  float64
	// Top 为绝对贡献最大的至多 3 个历史商品及其带符号贡献。
	Top []Contribution
	// Other 为其余历史商品贡献之和；Top 贡献之和加 Other 等于 Score。
	Other float64
}
