# KD-Tree in Go

This package provides a simple implementation of a KD-Tree in Go. A KD-Tree is a binary tree structure which supports efficient range and nearest neighbor queries for points in k-dimensional space.

## Features

- Build a KD-Tree from a slice of points.
- Find the k nearest neighbors to a given target point.
- Find all neighbors within a given radius of a target point.

## Usage

First, define the points you want to add to the KD-Tree:

```go
points := []Point{
    &PointBase{Vec: []float64{2, 3}},
    &PointBase{Vec: []float64{5, 4}},
    &PointBase{Vec: []float64{9, 6}},
    &PointBase{Vec: []float64{4, 7}},
    &PointBase{Vec: []float64{8, 1}},
    &PointBase{Vec: []float64{7, 2}},
}
```

Then, build the KD-Tree:

```go
tree := buildKDTree(points, 0)
```

To find the k nearest neighbors to a target point:

```go
target := &PointBase{Vec: []float64{9, 2}}
neighbors := tree.kNearestNeighbors(target, 3)
```

To find all neighbors within a given radius of a target point:

```go
radius := 5.0
neighbors := tree.neighborsWithinRadius(target, radius)
```

Note
This implementation of a KD-Tree uses the Euclidean distance to measure the distance between points. The points are represented as slices of float64 values, allowing for points in n-dimensional space.