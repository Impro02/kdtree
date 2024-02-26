package kdtree

import (
	"math/rand"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func TestKNearestNeighbors(t *testing.T) {

	points := []Point{
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

	kdTree := BuildKDTree(points, 0)

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
}

func TestNeighborsWithinRadius(t *testing.T) {

	points := []Point{
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

	kdTree := BuildKDTree(points, 0)

	target := &PointBase{Vec: []float64{3, 3}}
	k := 5.0
	workers := 2

	neighbors := kdTree.NeighborsWithinRadius(target, k, workers)

	expectedPoint := []Point{
		&PointBase{Vec: []float64{3, 3}},
		&PointBase{Vec: []float64{2, 2}},
		&PointBase{Vec: []float64{1, 1}},
		&PointBase{Vec: []float64{6, 6}},
	}

	assert.Equal(t, expectedPoint, neighbors)
}

func TestNeighborsWithinRadiusLargeDataset(t *testing.T) {
	// Generate a large dataset
	points := make([]Point, 1000000)
	for i := range points {
		points[i] = &PointBase{Vec: []float64{rand.Float64() * 1000, rand.Float64() * 1000}}
	}

	// Create a KDTree with the points
	kdTree := BuildKDTree(points, 0)

	// Test NeighborsWithinRadius
	center := &PointBase{Vec: []float64{500, 500}}
	radius := 100.0
	start := time.Now()
	neighbors := kdTree.NeighborsWithinRadius(center, radius, 2)
	elapsed := time.Since(start)
	t.Logf("Found %d neighbors within radius %.2f of point %.2f in %s", len(neighbors), radius, center.Vector(), elapsed)

	// Add assertions as needed, for example:
	if len(neighbors) == 0 {
		t.Errorf("Expected to find neighbors, but found none")
	}
}
