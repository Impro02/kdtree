package kdtree

import (
	"container/heap"
	"math"
	"sort"
	"sync"
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

	axis := depth % len(points[0].Vector())
	median := n / 2

	// Use a selection algorithm to find the median
	nthElement(points, median, axis)

	node := &Node{
		Point: points[median],
		Axis:  axis,
	}

	// Use a WaitGroup to wait for the goroutines to finish
	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		node.Left = BuildKDTree(points[:median], depth+1)
	}()

	go func() {
		defer wg.Done()
		node.Right = BuildKDTree(points[median+1:], depth+1)
	}()

	// Wait for the goroutines to finish
	wg.Wait()

	return node
}

// nthElement rearranges the slice such that the element at the nth position is the one that would be in that position in a sorted sequence.
func nthElement(points []Point, n, axis int) {
	sort.Slice(points, func(i, j int) bool {
		return points[i].GetValue(axis) < points[j].GetValue(axis)
	})
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
	if node == nil {
		return nil
	}

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
		var leftNeighbors, rightNeighbors []Point
		var wg sync.WaitGroup
		wg.Add(2)

		go func() {
			defer wg.Done()
			leftNeighbors = node.Left.NeighborsWithinRadius(target, radius, numWorkers)
		}()

		go func() {
			defer wg.Done()
			rightNeighbors = node.Right.NeighborsWithinRadius(target, radius, numWorkers)
		}()

		wg.Wait()

		neighbors = append(neighbors, leftNeighbors...)
		neighbors = append(neighbors, rightNeighbors...)
	} else if target.GetValue(dim) < node.Point.GetValue(dim) {
		neighbors = append(neighbors, node.Left.NeighborsWithinRadius(target, radius, numWorkers)...)
	}

	// Sort the neighbors from closest to farthest
	neighbors = mergeSort(neighbors, target)

	return neighbors
}

func mergeSort(arr []Point, target Point) []Point {
	if len(arr) <= 1 {
		return arr
	}

	middle := len(arr) / 2

	var left, right []Point
	done := make(chan bool)

	go func() {
		left = mergeSort(arr[:middle], target)
		done <- true
	}()

	right = mergeSort(arr[middle:], target)
	<-done

	return merge(left, right, target)
}

func merge(left, right []Point, target Point) (result []Point) {
	result = make([]Point, len(left)+len(right))

	i := 0
	for len(left) > 0 && len(right) > 0 {
		if target.Distance(left[0]) <= target.Distance(right[0]) {
			result[i] = left[0]
			left = left[1:]
		} else {
			result[i] = right[0]
			right = right[1:]
		}
		i++
	}

	for j := 0; j < len(left); j++ {
		result[i] = left[j]
		i++
	}
	for j := 0; j < len(right); j++ {
		result[i] = right[j]
		i++
	}

	return
}
