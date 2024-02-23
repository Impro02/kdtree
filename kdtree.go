package kdtree

import (
	"math"
	"sort"
)

type Point struct {
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
		return points[i].Vec[depth%len(points[i].Vec)] < points[j].Vec[depth%len(points[j].Vec)]
	})

	median := n / 2

	return &Node{
		Point: points[median],
		Left:  BuildKDTree(points[:median], depth+1),
		Right: BuildKDTree(points[median+1:], depth+1),
		Axis:  depth % len(points[0].Vec),
	}
}

func distance(a, b Point) float64 {
	sum := 0.0
	for i := range a.Vec {
		diff := a.Vec[i] - b.Vec[i]
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

		if node.Left != nil && (len(neighbors) < k || math.Abs(target.Vec[node.Axis]-node.Point.Vec[node.Axis]) < distance(target, neighbors[k-1])) {
			search(node.Left)
		}

		if node.Right != nil && (len(neighbors) < k || math.Abs(target.Vec[node.Axis]-node.Point.Vec[node.Axis]) < distance(target, neighbors[k-1])) {
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
		if target.Vec[node.Axis] < node.Point.Vec[node.Axis] {
			near = node.Left
			far = node.Right
		} else {
			near = node.Right
			far = node.Left
		}

		search(near)

		if d <= radius || math.Abs(target.Vec[node.Axis]-node.Point.Vec[node.Axis]) <= radius {
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
