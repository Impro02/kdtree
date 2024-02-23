package kdtree

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestKNearestNeighbors(t *testing.T) {

	points := []Point{
		{Vec: []float64{6, 6}},
		{Vec: []float64{1, 1}},
		{Vec: []float64{2, 2}},
		{Vec: []float64{10, 10}},
		{Vec: []float64{43, 43}},
		{Vec: []float64{9, 9}},
		{Vec: []float64{21, 21}},
		{Vec: []float64{3, 3}},
		{Vec: []float64{22, 22}},
		{Vec: []float64{40, 40}},
		{Vec: []float64{41, 41}},
		{Vec: []float64{20, 20}},
		{Vec: []float64{42, 42}},
		{Vec: []float64{100, 100}},
	}

	kdTree := BuildKDTree(points, 0)

	target := Point{Vec: []float64{3, 3}}
	k := 5

	neighbors := kdTree.kNearestNeighbors(target, k)

	expectedPoint := []Point{
		{Vec: []float64{3, 3}},
		{Vec: []float64{2, 2}},
		{Vec: []float64{1, 1}},
		{Vec: []float64{6, 6}},
		{Vec: []float64{9, 9}},
	}

	assert.Equal(t, expectedPoint, neighbors)
}

func TestNeighborsWithinRadius(t *testing.T) {

	points := []Point{
		{Vec: []float64{6, 6}},
		{Vec: []float64{1, 1}},
		{Vec: []float64{2, 2}},
		{Vec: []float64{10, 10}},
		{Vec: []float64{43, 43}},
		{Vec: []float64{9, 9}},
		{Vec: []float64{21, 21}},
		{Vec: []float64{3, 3}},
		{Vec: []float64{22, 22}},
		{Vec: []float64{40, 40}},
		{Vec: []float64{41, 41}},
		{Vec: []float64{20, 20}},
		{Vec: []float64{42, 42}},
		{Vec: []float64{100, 100}},
	}

	kdTree := BuildKDTree(points, 0)

	target := Point{Vec: []float64{3, 3}}
	k := 5.0

	neighbors := kdTree.neighborsWithinRadius(target, k)

	expectedPoint := []Point{
		{Vec: []float64{3, 3}},
		{Vec: []float64{2, 2}},
		{Vec: []float64{1, 1}},
		{Vec: []float64{6, 6}},
	}

	assert.Equal(t, expectedPoint, neighbors)
}
