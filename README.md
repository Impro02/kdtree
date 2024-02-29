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

## K Nearest Neighbors (KNN)

The `KNN` function can be used to find the k nearest neighbors to a given target point. Here's an example of how to use it:


```go
target := &PointBase{Vec: []float64{9, 2}}
k := 5
neighbors := tree.KNN(target, k)
```

In this example, target is the point for which we want to find the nearest neighbors, k is the number of neighbors we want to find.

## Search Radius
The `SearchRadius` function can be used to find all neighbors within a given radius of a target point. Here's an example of how to use it:

```go
target := &PointBase{Vec: []float64{9, 2}}
radius := 5.0
neighbors := tree.SearchRadius(target, radius)
```

In this example, target is the point for which we want to find the neighbors, and radius is the distance within which we want to find neighbors.

Note: This implementation of a KD-Tree uses the Euclidean distance to measure the distance between points. The points are represented as slices of float64 values, allowing for points in n-dimensional space.