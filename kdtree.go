package kdtree

import (
	"math"
	"sort"
)

type KDNode struct {
    point     *Point
	points    []*Point
    axis      int
    left      *KDNode
    right     *KDNode
}

type KDTree struct {
    root     *KDNode
    leafSize int
}

type Point struct {
	Vect []float64
}


func NewKDTree(points []*Point, leafSize int) *KDTree {
    var recFunc func(int, []*Point) *KDNode
    recFunc = func(axis int, points []*Point) *KDNode {
        if len(points) <= leafSize {
            return &KDNode{points: points}
        }

        sort.Slice(points, func(i, j int) bool {
            return points[i].Vect[axis] < points[j].Vect[axis]
        })

        median := len(points) / 2

        return &KDNode{
            point: points[median],
            axis:  axis,
            left:  recFunc((axis+1)%len(points[0].Vect), points[:median]),
            right: recFunc((axis+1)%len(points[0].Vect), points[median+1:]),
        }
    }

    root := recFunc(0, points)
    return &KDTree{root: root, leafSize: leafSize}
}

func (tree *KDTree) Insert(point *Point) {
    var recFunc func(node *KDNode) *KDNode
    recFunc = func(node *KDNode) *KDNode {
        if node == nil {
            return &KDNode{point: point}
        }
        dim := node.axis
        if point.Vect[dim] < node.point.Vect[dim] {
            node.left = recFunc(node.left)
        } else {
            node.right = recFunc(node.right)
        }
        return node
    }
    tree.root = recFunc(tree.root)
}

func (tree *KDTree) Nearest(point *Point, radius float64) []*Point {
    var recFunc func(node *KDNode) []*Point
    recFunc = func(node *KDNode) []*Point {
        if node == nil {
            return []*Point{}
        }

        if len(node.points) > 0 {
            // We're at a leaf node, so check all points
            var result []*Point
            for _, p := range node.points {
                if distance(point, p) <= radius {
                    result = append(result, p)
                }
            }
            return result
        }

        dim := node.axis
        nextBranch := node.left
        oppositeBranch := node.right

        if point.Vect[dim] > node.point.Vect[dim] {
            nextBranch = node.right
            oppositeBranch = node.left
        }

        best := recFunc(nextBranch)

        if math.Abs(point.Vect[dim]-node.point.Vect[dim]) < radius {
            best = append(best, recFunc(oppositeBranch)...)
        }

        return best
    }
    return recFunc(tree.root)
}

func distance(p1, p2 *Point) float64 {
	var ret float64
	for i := 0; i < len(p1.Vect); i++ {
		tmp := p1.Vect[i] - p2.Vect[i]
		ret += tmp * tmp
	}
	return math.Sqrt(ret)
}