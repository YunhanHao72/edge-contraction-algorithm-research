#include <vector>
#include <array>
#include <map>
#include <cassert>
#include <tuple>
#include <queue>
#include <iostream>
#include <fstream>
#include <random>

#include <partition.hxx>

namespace lookahead {

template<typename VALUE_TYPE>
class DynamicGraph
{
public:
    DynamicGraph(size_t n) : adjacency_(n) {}

    bool edgeExists(size_t a, size_t b) const
    {
        return adjacency_[a].count(b);
    }

    std::map<size_t, VALUE_TYPE> const& getAdjacentVertices(size_t v) const
    {
        return adjacency_[v];
    }

    VALUE_TYPE getEdgeWeight(size_t a, size_t b) const
    {
        return adjacency_[a].at(b);
    }

    void removeVertex(size_t v)
    {
        for (auto& p : adjacency_[v])
            adjacency_[p.first].erase(v);
        adjacency_[v].clear();
    }

    void updateEdgeWeight(size_t a, size_t b, VALUE_TYPE w)
    {
        adjacency_[a][b] += w;
        adjacency_[b][a] += w;
    }

private:
    std::vector<std::map<size_t, VALUE_TYPE>> adjacency_;
};

template<typename VALUE_TYPE>
struct Entry 
{
    Entry(double _prio, size_t _u, size_t _v, VALUE_TYPE _w)
    {
        if (_u > _v)
            std::swap(_u, _v);
        prio = _prio;
        u = _u;
        v = _v;
        w = _w;
    }

    double      prio;
    size_t      u, v;
    VALUE_TYPE  w;

    bool operator<(Entry const& other) const {
        if (prio != other.prio) return prio < other.prio;
        if (w != other.w) return w < other.w;
        if (u != other.u) return u > other.u;
        return v > other.v;
    }
};

template<typename VALUE_TYPE>
std::pair<VALUE_TYPE, std::vector<size_t>>
greedy_joining_lookahead(size_t n, std::vector<std::array<size_t, 2>> edges, std::vector<VALUE_TYPE> edge_values)
{
    assert(edges.size() == edge_values.size());
    constexpr double EPSILON_NOISE = 1e-9;
    constexpr double INF_APPROX = 1e20;
    constexpr double alpha = 0.3;
    std::uniform_real_distribution<double> noise_dist(-EPSILON_NOISE, EPSILON_NOISE);
    std::random_device rd;
    std::mt19937 rng(rd());

    DynamicGraph<VALUE_TYPE> graph(n);
    std::priority_queue<Entry<VALUE_TYPE>> queue;
    VALUE_TYPE value_of_cost = 0;

    std::vector<VALUE_TYPE> noisy_edge_values = edge_values;
    for (size_t i = 0; i < edges.size(); ++i)
    {
        size_t u = edges[i][0];
        size_t v = edges[i][1];
        noisy_edge_values[i] += noise_dist(rng);
        VALUE_TYPE w = noisy_edge_values[i];
        graph.updateEdgeWeight(u, v, w);
    }

    auto compute_reg = [&](size_t u, size_t v) -> double {
        double reg = 0.0;
        auto const& nbr_u = graph.getAdjacentVertices(u);
        auto const& nbr_v = graph.getAdjacentVertices(v);
        auto it_u = nbr_u.begin();
        auto it_v = nbr_v.begin();
        while (it_u != nbr_u.end() && it_v != nbr_v.end())
        {
            if (it_u->first == v) { ++it_u; continue; }
            if (it_v->first == u) { ++it_v; continue; }

            if (it_u->first == it_v->first)
            {
                double w_uk = it_u->second;
                double w_vk = it_v->second;
                if (w_uk * w_vk < 0)
                    reg += std::min(std::abs(w_uk), std::abs(w_vk));
                ++it_u;
                ++it_v;
            }
            else if (it_u->first < it_v->first)
            {
                ++it_u;
            }
            else
            {
                ++it_v;
            }
        }
        return reg;
    };

    for (size_t i = 0; i < edges.size(); ++i)
    {
        size_t u = edges[i][0];
        size_t v = edges[i][1];
        VALUE_TYPE w = noisy_edge_values[i];
        if (w > 0)
        {
            double reg = compute_reg(u, v);
            double prio = (reg > 0) ? (double(w) / reg) : INF_APPROX;
            queue.emplace(prio, u, v, w);
        }
    }

    andres::Partition<size_t> partition(n);

    while (!queue.empty())
    {
        auto data = queue.top();
        queue.pop();

        if (!graph.edgeExists(data.u, data.v) || data.w != graph.getEdgeWeight(data.u, data.v))
            continue;

        std::cout << "\rNumber of clusters: " << partition.numberOfSets() << "    " << std::flush;

        auto stable_vertex = data.u;
        auto merge_vertex = data.v;

        value_of_cost += data.w;

        partition.merge(stable_vertex, merge_vertex);

        for (auto& neighbor : graph.getAdjacentVertices(merge_vertex))
        {
            if (neighbor.first == stable_vertex)
                continue;

            graph.updateEdgeWeight(stable_vertex, neighbor.first, neighbor.second);

            VALUE_TYPE new_weight = graph.getEdgeWeight(stable_vertex, neighbor.first);
            if (new_weight > 0)
            {
                double reg = compute_reg(stable_vertex, neighbor.first);
                double prio = (reg > 0) ? (double(new_weight) / reg) : INF_APPROX;
                queue.emplace(prio, stable_vertex, neighbor.first, new_weight);
            }
        }

        graph.removeVertex(merge_vertex);
    }

    std::cout << "\r                                      \r" << std::flush;

    std::vector<size_t> labeling(n);
    partition.elementLabeling(labeling.begin());
    return {value_of_cost, labeling};
}

}