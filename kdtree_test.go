package kdtree

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

var points = []Point{
	&PointBase{Vec: []float64{6, 6}},
	&PointBase{Vec: []float64{1, 1}},
	&PointBase{Vec: []float64{2, 2}},
	&PointBase{Vec: []float64{10, 10}},
	&PointBase{Vec: []float64{43, 43}},
	&PointBase{Vec: []float64{9, 9}},
	&PointBase{Vec: []float64{21, 21}},
	&PointBase{Vec: []float64{3, 3}},
	&PointBase{Vec: []float64{22, 22}},
	&PointBase{Vec: []float64{40, 40}},
	&PointBase{Vec: []float64{41, 41}},
	&PointBase{Vec: []float64{20, 20}},
	&PointBase{Vec: []float64{42, 42}},
	&PointBase{Vec: []float64{100, 100}},
}

var kdTree = BuildKDTree(points, 0)

func TestKNearestNeighbors(t *testing.T) {

	target := &PointBase{Vec: []float64{3, 3}}
	k := 5
	workers := 2

	neighbors := kdTree.KNearestNeighbors(target, k, workers)

	expectedPoint := []Point{
		&PointBase{Vec: []float64{3, 3}},
		&PointBase{Vec: []float64{2, 2}},
		&PointBase{Vec: []float64{1, 1}},
		&PointBase{Vec: []float64{6, 6}},
		&PointBase{Vec: []float64{9, 9}},
	}

	assert.Equal(t, expectedPoint, neighbors)

	target = &PointBase{Vec: []float64{42, 42}}

	neighbors = kdTree.KNearestNeighbors(target, k, workers)

	expectedPoint = []Point{
		&PointBase{Vec: []float64{42, 42}},
		&PointBase{Vec: []float64{41, 41}},
		&PointBase{Vec: []float64{43, 43}},
		&PointBase{Vec: []float64{40, 40}},
		&PointBase{Vec: []float64{22, 22}},
	}

	assert.Equal(t, expectedPoint, neighbors)
}

func TestNeighborsWithinRadius(t *testing.T) {

	target := &PointBase{Vec: []float64{3, 3}}
	k := 5.0 * 5.0
	workers := 2

	neighbors := kdTree.NeighborsWithinRadius(target, k, workers)

	expectedPoint := []Point{
		&PointBase{Vec: []float64{3, 3}},
		&PointBase{Vec: []float64{2, 2}},
		&PointBase{Vec: []float64{1, 1}},
		&PointBase{Vec: []float64{6, 6}},
	}

	assert.Equal(t, expectedPoint, neighbors)

	target = &PointBase{Vec: []float64{42, 42}}

	neighbors = kdTree.NeighborsWithinRadius(target, k, workers)

	expectedPoint = []Point{
		&PointBase{Vec: []float64{42, 42}},
		&PointBase{Vec: []float64{41, 41}},
		&PointBase{Vec: []float64{43, 43}},
		&PointBase{Vec: []float64{40, 40}},
	}

	assert.Equal(t, expectedPoint, neighbors)
}
