package itemcf

import (
	"math"
	"math/rand"
	"runtime"
	"sort"
	"sync/atomic"
	"syscall"
	"testing"
	"time"
)

const (
	scaleUsers   = 100000
	scaleItems   = 100000
	scaleRatings = 2000000
	scaleRefDay  = 10000
	heavyUsers   = 10
	heavyHistory = 20000
	hotItem      = scaleItems // 商品 100000
	hotRaters    = 50000
	queryUsers   = 100
)

// genScaleRatings 生成确定性规模数据：10 名用户各 20000 条评分，
// 1 件热门商品被 50000 名用户评分，评分取值 1..5 不全相同。
func genScaleRatings() []Rating {
	rng := rand.New(rand.NewSource(42))
	ratings := make([]Rating, 0, scaleRatings)
	seen := make(map[int64]struct{}, scaleRatings)
	add := func(u, i int) bool {
		key := int64(u)<<32 | int64(i)
		if _, ok := seen[key]; ok {
			return false
		}
		seen[key] = struct{}{}
		ratings = append(ratings, Rating{
			UserID: u, ItemID: i,
			Score: 1 + rng.Intn(5),
			Day:   scaleRefDay - rng.Intn(366),
		})
		return true
	}
	// 10 名长历史用户，各 20000 条评分。
	for u := 1; u <= heavyUsers; u++ {
		for cnt := 0; cnt < heavyHistory; {
			if add(u, 1+rng.Intn(scaleItems)) {
				cnt++
			}
		}
	}
	// 1 件热门商品被 50000 名用户评分。
	for u := heavyUsers + 1; u <= heavyUsers+hotRaters; u++ {
		add(u, hotItem)
	}
	// 随机填满至 2000000 条。
	for len(ratings) < scaleRatings {
		add(1+rng.Intn(scaleUsers), 1+rng.Intn(scaleItems))
	}
	return ratings
}

// ---------- 测试侧独立精确计算（直接从原始评分按定义计算） ----------

type uw struct {
	item int
	w    float64
}

type scaleRef struct {
	byUser [][]uw // 按用户ID索引
	byItem [][]uw // 按商品ID索引，item 字段存用户ID
	norm   []float64
	invN   []float64
	count  []int
}

func buildScaleRef(ratings []Rating) *scaleRef {
	sum := make([]int64, scaleUsers+1)
	cnt := make([]int64, scaleUsers+1)
	for _, rt := range ratings {
		sum[rt.UserID] += int64(rt.Score)
		cnt[rt.UserID]++
	}
	m := &scaleRef{
		byUser: make([][]uw, scaleUsers+1),
		byItem: make([][]uw, scaleItems+1),
		norm:   make([]float64, scaleItems+1),
		invN:   make([]float64, scaleItems+1),
		count:  make([]int, scaleItems+1),
	}
	for _, rt := range ratings {
		mean := float64(sum[rt.UserID]) / float64(cnt[rt.UserID])
		w := (float64(rt.Score) - mean) * math.Pow(2, -float64(scaleRefDay-rt.Day)/30)
		m.byUser[rt.UserID] = append(m.byUser[rt.UserID], uw{rt.ItemID, w})
		m.byItem[rt.ItemID] = append(m.byItem[rt.ItemID], uw{rt.UserID, w})
		m.norm[rt.ItemID] += w * w
		m.count[rt.ItemID]++
	}
	for i := range m.norm {
		m.norm[i] = math.Sqrt(m.norm[i])
		if m.norm[i] != 0 {
			m.invN[i] = 1 / m.norm[i]
		}
	}
	return m
}

type refEntry struct {
	item  int
	score float64
}

// fullRanking 独立计算用户全部候选的精确得分并返回完整排名。
func (m *scaleRef) fullRanking(userID int) ([]refEntry, float64, map[int]float64) {
	hist := m.byUser[userID]
	histW := make(map[int]float64, len(hist))
	rated := make([]bool, scaleItems+1)
	sumAbs := 0.0
	for _, e := range hist {
		histW[e.item] = e.w
		rated[e.item] = true
		sumAbs += math.Abs(e.w)
	}
	num := make([]float64, scaleItems+1)
	if sumAbs > 0 {
		s := make([]float64, scaleUsers+1)
		sSeen := make([]bool, scaleUsers+1)
		var sTouched []int
		for i, wui := range histW {
			invNi := m.invN[i]
			if wui == 0 || invNi == 0 {
				continue
			}
			for _, ir := range m.byItem[i] { // ir.item 是用户 v
				if ir.w == 0 {
					continue
				}
				v := ir.item
				if !sSeen[v] {
					sSeen[v] = true
					sTouched = append(sTouched, v)
				}
				s[v] += wui * ir.w * invNi
			}
		}
		for _, v := range sTouched {
			sv := s[v]
			if sv == 0 {
				continue
			}
			for _, e := range m.byUser[v] {
				j := e.item
				if rated[j] || e.w == 0 {
					continue
				}
				num[j] += sv * e.w * m.invN[j]
			}
		}
	}
	denom := sumAbs
	entries := make([]refEntry, 0, scaleItems-len(hist))
	for item := 1; item <= scaleItems; item++ {
		if rated[item] {
			continue
		}
		score := 0.0
		if denom > 0 {
			score = num[item] / (denom * math.Log(2+float64(m.count[item])))
		}
		entries = append(entries, refEntry{item: item, score: score})
	}
	sort.Slice(entries, func(a, b int) bool {
		if entries[a].score != entries[b].score {
			return entries[a].score > entries[b].score
		}
		return entries[a].item < entries[b].item
	})
	return entries, sumAbs, histW
}

// refContribs 独立计算候选 j 的贡献解释。
func (m *scaleRef) refContribs(j int, histW map[int]float64, sumAbs, score float64) ([]Contribution, float64) {
	dot := map[int]float64{}
	for _, ir := range m.byItem[j] { // ir.item 是用户 v
		if ir.w == 0 {
			continue
		}
		for _, e := range m.byUser[ir.item] {
			if _, ok := histW[e.item]; ok {
				dot[e.item] += ir.w * e.w
			}
		}
	}
	denom := m.norm[j] * sumAbs * math.Log(2+float64(m.count[j]))
	var contribs []Contribution
	for i, d := range dot {
		ni := m.norm[i]
		if ni == 0 {
			continue
		}
		if v := histW[i] * d / (ni * denom); v != 0 {
			contribs = append(contribs, Contribution{ItemID: i, Value: v})
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
	return top, score - topSum
}

// ---------- 规模测试 ----------

func TestScale(t *testing.T) {
	ratings := genScaleRatings()
	if len(ratings) != scaleRatings {
		t.Fatalf("评分数不符: got %d", len(ratings))
	}
	// 校验长尾分布与评分多样性。
	userCnt := map[int]int{}
	itemCnt := map[int]int{}
	scoreSeen := map[int]bool{}
	for _, rt := range ratings {
		userCnt[rt.UserID]++
		itemCnt[rt.ItemID]++
		scoreSeen[rt.Score] = true
	}
	heavy := 0
	for _, c := range userCnt {
		if c >= heavyHistory {
			heavy++
		}
	}
	if heavy < heavyUsers {
		t.Fatalf("长历史用户不足: got %d, want >= %d", heavy, heavyUsers)
	}
	if itemCnt[hotItem] < hotRaters {
		t.Fatalf("热门商品评分人数不足: got %d", itemCnt[hotItem])
	}
	if len(scoreSeen) < 2 {
		t.Fatalf("评分全部相同，不符合要求")
	}

	// 查询用户：全部 10 名长历史用户 + 90 名随机用户。
	rng := rand.New(rand.NewSource(7))
	queries := make([]int, 0, queryUsers)
	for u := 1; u <= heavyUsers; u++ {
		queries = append(queries, u)
	}
	picked := map[int]bool{}
	for len(queries) < queryUsers {
		u := heavyUsers + 1 + rng.Intn(scaleUsers-heavyUsers)
		if !picked[u] {
			picked[u] = true
			queries = append(queries, u)
		}
	}

	// 计时与内存采样：构建 + 100 个不同用户的完整推荐（含贡献解释）。
	// 先触发 GC，排除数据生成阶段垃圾对内存峰值测量的干扰。
	runtime.GC()
	var peak int64
	stop := make(chan struct{})
	go func() {
		var ms runtime.MemStats
		for {
			select {
			case <-stop:
				return
			default:
				runtime.ReadMemStats(&ms)
				for {
					p := atomic.LoadInt64(&peak)
					if int64(ms.HeapAlloc) <= p || atomic.CompareAndSwapInt64(&peak, p, int64(ms.HeapAlloc)) {
						break
					}
				}
				time.Sleep(time.Millisecond)
			}
		}
	}()
	start := time.Now()
	cpuStart := selfCPU()
	rec := Build(ratings, scaleRefDay)
	results := make(map[int][]Recommendation, queryUsers)
	for _, u := range queries {
		results[u] = rec.Recommend(u, nil)
	}
	elapsed := time.Since(start)
	cpuElapsed := selfCPU() - cpuStart
	close(stop)
	t.Logf("构建+100次推荐: CPU时间 %v, 墙钟时间 %v, 峰值堆内存 %.1f MiB",
		cpuElapsed, elapsed, float64(atomic.LoadInt64(&peak))/float64(uint64(1)<<20))
	if cpuElapsed > 60*time.Second {
		t.Fatalf("CPU时间超限: %v > 60s", cpuElapsed)
	}
	if atomic.LoadInt64(&peak) > 512<<20 {
		t.Fatalf("峰值内存超限: %.1f MiB > 512 MiB", float64(atomic.LoadInt64(&peak))/float64(uint64(1)<<20))
	}
	for _, u := range queries {
		if len(results[u]) != 20 {
			t.Fatalf("用户 %d 推荐条数不足 20: %d", u, len(results[u]))
		}
	}

	// 核对抽取用户（3 名长历史用户 + 2 名普通用户）的完整候选排名。
	ref := buildScaleRef(ratings)
	sampled := []int{1, 2, 3, queries[10], queries[50]}
	for _, u := range sampled {
		entries, sumAbs, histW := ref.fullRanking(u)
		if len(entries) < 20 {
			t.Fatalf("用户 %d 候选不足 20", u)
		}
		got := results[u]
		for k := 0; k < 20; k++ {
			if got[k].ItemID != entries[k].item {
				t.Fatalf("用户 %d 第 %d 名不符: got %d, want %d", u, k, got[k].ItemID, entries[k].item)
			}
			if math.Abs(got[k].Score-entries[k].score) > 1e-9 {
				t.Fatalf("用户 %d 商品 %d 得分误差过大: got %v, want %v",
					u, got[k].ItemID, got[k].Score, entries[k].score)
			}
			// 核对贡献解释。
			top, other := ref.refContribs(entries[k].item, histW, sumAbs, entries[k].score)
			if len(got[k].Top) != len(top) {
				t.Fatalf("用户 %d 商品 %d 贡献条数不符: got %d, want %d",
					u, got[k].ItemID, len(got[k].Top), len(top))
			}
			for c := range top {
				if got[k].Top[c].ItemID != top[c].ItemID ||
					math.Abs(got[k].Top[c].Value-top[c].Value) > 1e-9 {
					t.Fatalf("用户 %d 商品 %d 第 %d 个贡献不符: got %+v, want %+v",
						u, got[k].ItemID, c, got[k].Top[c], top[c])
				}
			}
			if math.Abs(got[k].Other-other) > 1e-9 {
				t.Fatalf("用户 %d 商品 %d 其余贡献和不符: got %v, want %v",
					u, got[k].ItemID, got[k].Other, other)
			}
		}
		t.Logf("用户 %d 完整候选排名核对通过（候选共 %d 件）", u, len(entries))
	}
}

// selfCPU 返回当前进程已消耗的 CPU 时间（用户态 + 内核态）。
func selfCPU() time.Duration {
	var ru syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &ru); err != nil {
		return 0
	}
	return time.Duration(ru.Utime.Nano() + ru.Stime.Nano())
}
