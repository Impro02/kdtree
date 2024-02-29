package kdtree

import (
	"math/rand"
	"testing"

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

	kdTree := BuildKDTree(points, 0, 3)

	target := &PointBase{Vec: []float64{3, 3}}
	k := 5

	neighbors := kdTree.KNN(target, k)

	expectedPoint := []Point{
		&PointBase{Vec: []float64{9, 9}},
		&PointBase{Vec: []float64{6, 6}},
		&PointBase{Vec: []float64{1, 1}},
		&PointBase{Vec: []float64{2, 2}},
		&PointBase{Vec: []float64{3, 3}},
	}

	assert.Equal(t, expectedPoint, neighbors)

	target = &PointBase{Vec: []float64{42, 42}}

	neighbors = kdTree.KNN(target, k)

	expectedPoint = []Point{
		&PointBase{Vec: []float64{22, 22}},
		&PointBase{Vec: []float64{40, 40}},
		&PointBase{Vec: []float64{41, 41}},
		&PointBase{Vec: []float64{43, 43}},
		&PointBase{Vec: []float64{42, 42}},
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

	kdTree := BuildKDTree(points, 0, 3)

	target := &PointBase{Vec: []float64{3, 3}}
	k := 5.0

	neighbors := kdTree.SearchRadius(target, k)

	expectedPoint := []Point{
		&PointBase{Vec: []float64{6, 6}},
		&PointBase{Vec: []float64{1, 1}},
		&PointBase{Vec: []float64{2, 2}},
		&PointBase{Vec: []float64{3, 3}},
	}

	assert.Equal(t, expectedPoint, neighbors)

	target = &PointBase{Vec: []float64{42, 42}}

	neighbors = kdTree.SearchRadius(target, k)

	expectedPoint = []Point{
		&PointBase{Vec: []float64{42, 42}},
		&PointBase{Vec: []float64{43, 43}},
		&PointBase{Vec: []float64{40, 40}},
		&PointBase{Vec: []float64{41, 41}},
	}

	assert.Equal(t, expectedPoint, neighbors)
}

func TestKNearestNeighborsLargeDataset(t *testing.T) {
	// Seed the random number generator with a constant value.
	r := rand.New(rand.NewSource(42))

	// Define the centroids.
	centroids := []Point{
		&PointBase{Vec: []float64{0.1, 0.1}},
		&PointBase{Vec: []float64{0.1, 0.9}},
		&PointBase{Vec: []float64{0.5, 0.5}},
		&PointBase{Vec: []float64{0.9, 0.1}},
		&PointBase{Vec: []float64{0.9, 0.9}},
	}

	points := []Point{}
	for i := 0; i < 1000000; i++ {
		centroid := centroids[r.Intn(len(centroids))]
		x := centroid.GetValue(0) + r.NormFloat64()*0.1
		y := centroid.GetValue(1) + r.NormFloat64()*0.1
		points = append(points, &PointBase{Vec: []float64{x, y}})
	}

	kdTree := BuildKDTree(points, 0, 30)

	target := &PointBase{Vec: []float64{r.Float64(), r.Float64()}}
	k := 5

	neighbors := kdTree.KNN(target, k)

	expectedPoint := []Point{
		&PointBase{Vec: []float64{0.3856857299481051, 0.5419476199994974}},
		&PointBase{Vec: []float64{0.38542887388874814, 0.5417771692037563}},
		&PointBase{Vec: []float64{0.3854901841318027, 0.541756239560265}},
		&PointBase{Vec: []float64{0.38549966558989135, 0.5415744489834114}},
		&PointBase{Vec: []float64{0.38564315658887993, 0.5414216231963146}},
	}

	assert.Equal(t, expectedPoint, neighbors)

	target = &PointBase{Vec: []float64{r.Float64(), r.Float64()}}

	neighbors = kdTree.KNN(target, k)

	expectedPoint = []Point{
		&PointBase{Vec: []float64{0.7497493291422486, 0.3950725887601213}},
		&PointBase{Vec: []float64{0.7466782147046156, 0.40251297114855955}},
		&PointBase{Vec: []float64{0.7477548717536625, 0.4024430684518462}},
		&PointBase{Vec: []float64{0.750271368071364, 0.40175926458750155}},
		&PointBase{Vec: []float64{0.7490513219372369, 0.39989061448023705}},
	}

	assert.Equal(t, expectedPoint, neighbors)
}

func TestNeighborsWithinRadiusLargeDataset(t *testing.T) {
	// Seed the random number generator with a constant value.
	r := rand.New(rand.NewSource(42))

	// Define the centroids.
	centroids := []Point{
		&PointBase{Vec: []float64{0.1, 0.1}},
		&PointBase{Vec: []float64{0.1, 0.9}},
		&PointBase{Vec: []float64{0.5, 0.5}},
		&PointBase{Vec: []float64{0.9, 0.1}},
		&PointBase{Vec: []float64{0.9, 0.9}},
	}

	points := []Point{}
	for i := 0; i < 1000000; i++ {
		centroid := centroids[r.Intn(len(centroids))]
		x := centroid.GetValue(0) + r.NormFloat64()*0.1
		y := centroid.GetValue(1) + r.NormFloat64()*0.1
		points = append(points, &PointBase{Vec: []float64{x, y}})
	}

	kdTree := BuildKDTree(points, 0, 30)

	target := &PointBase{Vec: []float64{r.Float64(), r.Float64()}}
	k := 0.0005

	neighbors := kdTree.SearchRadius(target, k)

	expectedPoint := []Point{
		&PointBase{Vec: []float64{0.38564315658887993, 0.5414216231963146}},
		&PointBase{Vec: []float64{0.38549966558989135, 0.5415744489834114}},
		&PointBase{Vec: []float64{0.3854901841318027, 0.541756239560265}},
		&PointBase{Vec: []float64{0.38542887388874814, 0.5417771692037563}},
	}

	assert.Equal(t, expectedPoint, neighbors)

	target = &PointBase{Vec: []float64{r.Float64(), r.Float64()}}
	k = 0.005

	neighbors = kdTree.SearchRadius(target, k)

	expectedPoint = []Point{
		&PointBase{Vec: []float64{0.750084660506549, 0.3945275932480578}},
		&PointBase{Vec: []float64{0.7497493291422486, 0.3950725887601213}},
		&PointBase{Vec: []float64{0.7471632682870435, 0.3952894584674514}},
		&PointBase{Vec: []float64{0.751467324024117, 0.39559326843639386}},
		&PointBase{Vec: []float64{0.7441017049545757, 0.3994247485954816}},
		&PointBase{Vec: []float64{0.7490513219372369, 0.39989061448023705}},
		&PointBase{Vec: []float64{0.750271368071364, 0.40175926458750155}},
		&PointBase{Vec: []float64{0.7477548717536625, 0.4024430684518462}},
		&PointBase{Vec: []float64{0.7466782147046156, 0.40251297114855955}},
	}

	assert.Equal(t, expectedPoint, neighbors)
}

func BenchmarkNeighborsWithinRadius(b *testing.B) {
	// Seed the random number generator with a constant value.
	r := rand.New(rand.NewSource(42))

	// Generate a large dataset for the benchmark
	points := make([]Point, 1000000)
	for i := range points {
		points[i] = &PointBase{Vec: []float64{r.Float64(), r.Float64()}}
	}

	kdTree := BuildKDTree(points, 0, 30)
	target := &PointBase{Vec: []float64{0.5, 0.5}}
	radius := 0.1

	// Reset the timer to exclude the setup time
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		kdTree.SearchRadius(target, radius)
	}
}
