package kdtree

import (
	"container/heap"
	"math"
	"sort"
	"sync"
)

type Point interface {
	Vector() []float64
	Dim() int
	GetValue(i int) float64
	Distance(other Point) float64
	PlaneDistance(other Point, axis int) float64
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
	return ret
}

func (p *PointBase) PlaneDistance(other Point, axis int) float64 {
	return p.GetValue(axis) - other.GetValue(axis)
}

type PointBase struct {
	Vec []float64
}

type Node struct {
	Point  Point
	Points []Point
	Left   *Node
	Right  *Node
	Axis   int
}

type Result struct {
	Point    Point
	Distance float64
}

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

func BuildKDTree(points []Point, depth int, leafSize int) *Node {
	n := len(points)

	if n == 0 {
		return nil
	}

	// Determine the splitting axis
	axis := depth % len(points[0].Vector())

	// Pre-sort the points along the splitting dimension
	nthElement(points, axis)

	// Find the median
	median := n / 2

	if n <= leafSize {
		return &Node{
			Point:  points[median],
			Points: points,
			Axis:   axis,
		}
	}

	// Create a new node
	node := &Node{
		Point: points[median],
		Axis:  axis,
	}

	// Use a WaitGroup to wait for the goroutines to finish
	var wg sync.WaitGroup
	wg.Add(2)

	// Build the left subtree in a new goroutine
	go func() {
		defer wg.Done()
		node.Left = BuildKDTree(points[:median], depth+1, leafSize)
	}()

	// Build the right subtree in a new goroutine
	go func() {
		defer wg.Done()
		node.Right = BuildKDTree(points[median+1:], depth+1, leafSize)
	}()

	// Wait for the goroutines to finish
	wg.Wait()

	return node
}

func nthElement(points []Point, axis int) {
	sort.Slice(points, func(i, j int) bool {
		return points[i].GetValue(axis) < points[j].GetValue(axis)
	})
}

func (node *Node) KNN(target Point, k int, numWorkers int) []Point {
	// Create a max heap to store the k nearest neighbors
	h := &MaxHeap{}
	heap.Init(h)

	// Create a function that will be used to search the KD-Tree
	var search func(node *Node)
	search = func(node *Node) {
		if node == nil {
			return
		}

		// If the node is a leaf node
		if len(node.Points) > 0 {
			for _, point := range node.Points {
				distance := point.Distance(target)
				if h.Len() < k {
					heap.Push(h, Result{Point: point, Distance: distance})
				} else if top := (*h)[0]; distance < top.Distance {
					(*h)[0] = Result{Point: point, Distance: distance}
					heap.Fix(h, 0)
				}
			}
		} else {
			// If the node is not a leaf node
			distance := node.Point.Distance(target)

			if h.Len() < k {
				heap.Push(h, Result{Point: node.Point, Distance: distance})
			} else if top := (*h)[0]; distance < top.Distance {
				(*h)[0] = Result{Point: node.Point, Distance: distance}
				heap.Fix(h, 0)
			}

			// Determine which subtree to search first
			closeBranch, farBranch := node.Left, node.Right
			if target.PlaneDistance(node.Point, node.Axis) > 0 {
				closeBranch, farBranch = node.Right, node.Left
			}

			search(closeBranch)
			if h.Len() < k || math.Abs(distance) <= (*h)[0].Distance {
				search(farBranch)
			}
		}
	}

	// Start the search
	search(node)

	// Extract the k nearest neighbors from the heap
	neighbors := make([]Point, 0, k)
	for h.Len() > 0 {
		neighbors = append(neighbors, heap.Pop(h).(Result).Point)
	}

	return neighbors
}

func (node *Node) SearchRadius(target Point, radius float64) []Point {
	var result []Point
	node.searchRadius(target, radius, &result)
	return result
}

func (node *Node) searchRadius(target Point, radius float64, result *[]Point) {
	if node == nil {
		return
	}

	if len(node.Points) > 0 {
		for _, point := range node.Points {
			d := target.Distance(point)
			if d <= radius {
				*result = append(*result, point)
			}
		}
	} else {
		// Calculate the distance from the target point to the current node's point
		d := target.Distance(node.Point)

		// If the current node's point is within the radius, add it to the result
		if d <= radius {
			*result = append(*result, node.Point)
		}

		// Calculate the distance from the target point to the current node's splitting plane
		planeDistance := target.PlaneDistance(node.Point, node.Axis)

		// First, search the subtree on the same side as the target point
		if planeDistance < 0 {
			node.Left.searchRadius(target, radius, result)
		} else {
			node.Right.searchRadius(target, radius, result)
		}

		// If the splitting plane intersects the search sphere, also search the other subtree
		if planeDistance*planeDistance <= radius {
			if planeDistance < 0 {
				node.Right.searchRadius(target, radius, result)
			} else {
				node.Left.searchRadius(target, radius, result)
			}
		}
	}

}
