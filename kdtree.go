package kdtree

import (
	"math"
	"sort"
)

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

func (node *Node) kNearestNeighbors(target Point, k int) []Point {
	neighbors := make([]Point, 0, k)

	var search func(node *Node)

	search = func(node *Node) {
		if node == nil {
			return
		}

		d := distance(target, node.Point)

		if len(neighbors) < k || d < distance(target, neighbors[k-1]) {
			if len(neighbors) == k {
				neighbors = neighbors[:k-1] // Remove the farthest neighbor
			}
			neighbors = append(neighbors, node.Point)
			sort.Slice(neighbors, func(i, j int) bool {
				return distance(target, neighbors[i]) < distance(target, neighbors[j])
			})
		}

		if node.Left != nil && (len(neighbors) < k || math.Abs(target.GetValue(node.Axis)-node.Point.GetValue(node.Axis)) < distance(target, neighbors[k-1])) {
			search(node.Left)
		}

		if node.Right != nil && (len(neighbors) < k || math.Abs(target.GetValue(node.Axis)-node.Point.GetValue(node.Axis)) < distance(target, neighbors[k-1])) {
			search(node.Right)
		}
	}

	search(node)

	return neighbors
}

func (node *Node) neighborsWithinRadius(target Point, radius float64) []Point {
	neighbors := make([]Point, 0)

	var search func(node *Node)

	search = func(node *Node) {
		if node == nil {
			return
		}

		d := distance(target, node.Point)

		if d <= radius {
			neighbors = append(neighbors, node.Point)
		}

		var near, far *Node
		if target.GetValue(node.Axis) < node.Point.GetValue(node.Axis) {
			near = node.Left
			far = node.Right
		} else {
			near = node.Right
			far = node.Left
		}

		search(near)

		if d <= radius || math.Abs(target.GetValue(node.Axis)-node.Point.GetValue(node.Axis)) <= radius {
			search(far)
		}
	}

	search(node)

	// Sort the neighbors from closest to farthest
	sort.Slice(neighbors, func(i, j int) bool {
		return distance(target, neighbors[i]) < distance(target, neighbors[j])
	})

	return neighbors
}
