package kdtree

import (
	"container/heap"
	"math"
	"sort"
)

type MaxHeap []Result

func (h MaxHeap) Len() int           { return len(h) }
func (h MaxHeap) Less(i, j int) bool { return h[i].Distance > h[j].Distance }
func (h MaxHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *MaxHeap) Push(x interface{}) {
	*h = append(*h, x.(Result))
}

func (h *MaxHeap) Pop() interface{} {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

type Point interface {
	Vector() []float64
	Dim() int
	GetValue(i int) float64
	Distance(other Point) float64
}

func (p *PointBase) Vector() []float64 {
	return p.Vec
}

func (p *PointBase) Dim() int {
	return len(p.Vec)
}

func (p *PointBase) GetValue(dim int) float64 {
	return p.Vec[dim]
}

func (p *PointBase) Distance(other Point) float64 {
	var ret float64
	for i := 0; i < p.Dim(); i++ {
		tmp := p.GetValue(i) - other.GetValue(i)
		ret += tmp * tmp
	}
	return math.Sqrt(ret)
}

type PointBase struct {
	Vec []float64
}

type Node struct {
	Point Point
	Left  *Node
	Right *Node
	Axis  int
}

type Job struct {
	Node   *Node
	Target Point
}

type Result struct {
	Point    Point
	Distance float64
}

func (node *Node) flatten() []*Node {
	if node == nil {
		return nil
	}

	nodes := []*Node{node}

	if node.Left != nil {
		nodes = append(nodes, node.Left.flatten()...)
	}
	if node.Right != nil {
		nodes = append(nodes, node.Right.flatten()...)
	}

	return nodes
}

func worker(jobs <-chan Job, results chan<- Result) {
	for job := range jobs {
		d := distance(job.Target, job.Node.Point)
		results <- Result{Point: job.Node.Point, Distance: d}
	}
}

func createWorkerPool(numWorkers int, jobs <-chan Job, results chan<- Result) {
	for i := 0; i < numWorkers; i++ {
		go worker(jobs, results)
	}
}

func BuildKDTree(points []Point, depth int) *Node {
	n := len(points)

	if n == 0 {
		return nil
	}

	sort.Slice(points, func(i, j int) bool {
		return points[i].GetValue(depth%len(points[i].Vector())) < points[j].GetValue(depth%len(points[j].Vector()))
	})

	median := n / 2

	return &Node{
		Point: points[median],
		Left:  BuildKDTree(points[:median], depth+1),
		Right: BuildKDTree(points[median+1:], depth+1),
		Axis:  depth % len(points[0].Vector()),
	}
}

func distance(a, b Point) float64 {
	sum := 0.0
	for i := range a.Dim() {
		diff := a.GetValue(i) - b.GetValue(i)
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

func (node *Node) KNearestNeighbors(target Point, k int, numWorkers int) []Point {
	nodes := node.flatten() // Flatten the tree into a slice of nodes
	jobs := make(chan Job, len(nodes))
	results := make(chan Result, len(nodes))

	createWorkerPool(numWorkers, jobs, results)

	for _, node := range nodes {
		jobs <- Job{Node: node, Target: target}
	}
	close(jobs)

	h := &MaxHeap{}
	heap.Init(h)

	for i := 0; i < len(nodes); i++ {
		result := <-results
		if h.Len() < k {
			heap.Push(h, result)
		} else if top := (*h)[0]; result.Distance < top.Distance {
			heap.Pop(h)
			heap.Push(h, result)
		}
	}

	var neighbors []Point
	for h.Len() > 0 {
		neighbors = append(neighbors, heap.Pop(h).(Result).Point)
	}

	// Reverse the slice because heap.Pop gives the largest first
	for i := len(neighbors)/2 - 1; i >= 0; i-- {
		opp := len(neighbors) - 1 - i
		neighbors[i], neighbors[opp] = neighbors[opp], neighbors[i]
	}

	return neighbors
}

func (node *Node) NeighborsWithinRadius(target Point, radius float64, numWorkers int) []Point {
	if node == nil {
		return nil
	}

	var neighbors []Point
	if node.Point.Distance(target) <= radius {
		neighbors = append(neighbors, node.Point)
	}

	dim := node.Axis
	if math.Abs(target.GetValue(dim)-node.Point.GetValue(dim)) <= radius {
		neighbors = append(neighbors, node.Left.NeighborsWithinRadius(target, radius, numWorkers)...)
		neighbors = append(neighbors, node.Right.NeighborsWithinRadius(target, radius, numWorkers)...)
	} else if target.GetValue(dim) < node.Point.GetValue(dim) {
		neighbors = append(neighbors, node.Left.NeighborsWithinRadius(target, radius, numWorkers)...)
	} else {
		neighbors = append(neighbors, node.Right.NeighborsWithinRadius(target, radius, numWorkers)...)
	}

	// Sort the neighbors from closest to farthest
	sort.Slice(neighbors, func(i, j int) bool {
		return target.Distance(neighbors[i]) < target.Distance(neighbors[j])
	})

	return neighbors
}
