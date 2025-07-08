#include <vector>
#include <array>
#include <map>
#include <cassert>
#include <tuple>
#include <queue>
#include <iostream>
#include <random>

#include <partition.hxx>


/**
 * This class implements an undirected edge weighted graph data structure.
 * The number of nodes n is fixed. Edges can be inserted and removed in time log(n).
 */
template<typename VALUE_TYPE>
class DynamicGraph
{
public:
    /**
     * Initialize the data structure by specifying the number of nodes n
     */
    DynamicGraph(size_t n) :
        // create an empty adjacency vector of length n
        adjacency_(n)
    {}

    /**
     * check if the edge {a, b} exists
     */
    bool edgeExists(size_t a, size_t b) const
    {
        return adjacency_[a].count(b);
    }

    /**
     * Get a constant reference to the adjacent vertices of vertex v
     */
    std::map<size_t, VALUE_TYPE> const& getAdjacentVertices(size_t v) const
    {
        return adjacency_[v];
    }

    /**
     * Get the weight of the edge {a, b}
     */
    VALUE_TYPE getEdgeWeight(size_t a, size_t b) const
    {
        return adjacency_[a].at(b);
    }

    /**
     * Remove all edges incident to the vertex v
     */
    void removeVertex(size_t v)
    {
        // for each vertex p that is incident to v, remove v from the adjacency of p
        for (auto& p : adjacency_[v])
            adjacency_[p.first].erase(v);
        // clear the adjacency of v
        adjacency_[v].clear();
    }

    /**
     * Increase the weight of the edge {a, b} by w.
     * In particular this function can be used to insert a new edge.
     */
    void updateEdgeWeight(size_t a, size_t b, VALUE_TYPE w)
    {
        adjacency_[a][b] += w;
        adjacency_[b][a] += w;
    }

private:
    // data structure that contains one map for each vertex v and maps from
    // the neighbors w of v to the weight of the edge {v, w}. This data
    // structure is kept symmetric as it models undirected graph, i.e.,
    // at all times it holds that adjacency[v][w] = adjacency_[w][v]
    std::vector<std::map<size_t, VALUE_TYPE>> adjacency_;
};


/**
 * This struct implements an edge of a graph consisting of its two end vertices
 * a and b (where by convention a is the vertex with the smaller index) and a weight w.
 * This struct also implements a comparison operator for comparing two edges based on their weight.
 */
template<typename VALUE_TYPE>
struct Edge
{
    /**
     * Initialize and edge {a, b} with weight w
     */
    Edge(size_t _a, size_t _b, VALUE_TYPE _w)
    {
        if (_a > _b)
            std::swap(_a, _b);

        a = _a;
        b = _b;
        w = _w;
    }

    size_t a;
    size_t b;
    VALUE_TYPE w;

    /**
     * Compare this edge to another edge based on their weights
     */
    // bool operator <(Edge const& other) const
    // {
    //     return w < other.w;
    // }
    bool operator <(Edge const& other) const
    {
        if (w != other.w)
            return w < other.w;
        if (a != other.a)
            return a > other.a;
        return b > other.b;
    }
};



template<typename VALUE_TYPE>
std::pair<VALUE_TYPE, std::vector<size_t>>
greedy_joining(size_t n, std::vector<std::array<size_t, 2>> edges, std::vector<VALUE_TYPE> edge_values)
{
    assert (edges.size() == edge_values.size());
    constexpr double EPSILON_NOISE = 1e-9;
    std::uniform_real_distribution<double> noise_dist(-EPSILON_NOISE, EPSILON_NOISE);
    std::random_device rd;
    std::mt19937 rng(rd()); 
    DynamicGraph<VALUE_TYPE> graph(n);
    std::priority_queue<Edge<VALUE_TYPE>> queue;
    VALUE_TYPE value_of_cost = 0;

    // initialize the graph by inserting all edges with their respective values
    std::vector<VALUE_TYPE> noisy_weights = edge_values;
    for (size_t i = 0; i < edges.size(); ++i)
    {
        noisy_weights[i] += noise_dist(rng);
        size_t a = edges[i][0];
        size_t b = edges[i][1];
        VALUE_TYPE w = noisy_weights[i];
        graph.updateEdgeWeight(a, b, w);
        // If the edge has positive weight, add it to the queue for possibly joining this edge
        if (w > 0)
        {
            queue.push({a, b, w});
        }
    }

    // initialize the partition where originally all vertices are in their own cluster
    andres::Partition<size_t> partition(n);

    // main loop of the greedy joining algorithm
    while (!queue.empty())
    {
        // get the top edge from the queue, i.e. the edge with the most negative value
        auto edge = queue.top();
        queue.pop();

        // check if this edge is outdated (i.e. it is no longer part of the graph or has different costs)
        if (!graph.edgeExists(edge.a, edge.b) || edge.w != graph.getEdgeWeight(edge.a, edge.b))
            continue;

        // print the current number of clusters
        std::cout << "\rNumber of clusters: " << partition.numberOfSets() << "    " << std::flush;

        // select one of the two vertices a and b that should be kept and the other vertex is
        // merged into the vertex that is kept. Do this such that the vertex with the larger
        // adjacency is the vertex that is kept.
        auto stable_vertex = edge.a;
        auto merge_vertex = edge.b;
        if (graph.getAdjacentVertices(stable_vertex).size() < graph.getAdjacentVertices(merge_vertex).size())
            std::swap(stable_vertex, merge_vertex);

        value_of_cost += edge.w;

        // merge the clusters associated with the two vertices in the partition
        partition.merge(stable_vertex, merge_vertex);

        // update the graph by adding and edge from the stable_vertex to all neighbors of
        // the merge_vertex and updating the value of such an edge if it already exists
        for (auto& neighbor : graph.getAdjacentVertices(merge_vertex))
        {
            if (neighbor.first == stable_vertex)
                continue;

            graph.updateEdgeWeight(stable_vertex, neighbor.first, neighbor.second);

            // if the value of the new weight is positive, add it to the queue as a
            // candidate for joining in a future iteration
            VALUE_TYPE new_weight = graph.getEdgeWeight(stable_vertex, neighbor.first);
            if (new_weight > 0)
                queue.push({stable_vertex, neighbor.first, new_weight});
        }
        // remove all edges incident to the merge vertex from the graph
        graph.removeVertex(merge_vertex);
    }
    std::cout << "\r                                      \r" << std::flush;


    // return a node labeling and the value of the computed solution
    std::vector<size_t> labeling(n);
    partition.elementLabeling(labeling.begin());
    return {value_of_cost, labeling};
}