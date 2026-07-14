#ifndef XCGD_QUADTREE_H
#define XCGD_QUADTREE_H

#include <stdio.h>

#include <iostream>

#include "quadrant.h"

namespace xcgd {

class Quadtree {
 public:
  Quadtree() {
    // Make a quadtree with a single quadrant
    std::vector<Quadrant> vec(1);
    vec[0].level = 0;
    vec[0].info = 0;
    vec[0].tag = 0;
    vec[0].x = 0;
    vec[0].y = 0;
    quadrants = std::make_shared<QuadrantArray>(std::move(vec));
  }

  /**
   * @brief Get the number of quadrants
   *
   * @return int
   */
  int size() const { return quadrants->size(); }

  /**
   * @brief Write to VTK
   *
   * @param filename The VTK filename
   */
  void to_vtk(std::string filename) const {
    QuadrantArray& quads = *quadrants;

    FILE* fp = std::fopen(filename.c_str(), "w");

    if (fp) {
      std::fprintf(fp, "# vtk DataFile Version 3.0\n");
      std::fprintf(fp, "vtk output\nASCII\n");
      std::fprintf(fp, "DATASET UNSTRUCTURED_GRID\n");

      // Write out the points
      int size = quads.size();
      std::fprintf(fp, "POINTS %d float\n", 4 * size);

      const std::int32_t hmax = 1 << Quadrant::MAX_LEVEL;

      for (int k = 0; k < size; k++) {
        const std::int32_t h = quads[k].get_size();

        // Set the point
        for (int jj = 0; jj < 2; jj++) {
          for (int ii = 0; ii < 2; ii++) {
            double u = 1.0 * (quads[k].x + ii * h) / hmax;
            double v = 1.0 * (quads[k].y + jj * h) / hmax;
            double w = 0.0;

            std::fprintf(fp, "%e %e %e\n", u, v, w);
          }
        }
      }

      // Write out the cell values
      std::fprintf(fp, "\nCELLS %d %d\n", size, 5 * size);
      for (int k = 0; k < size; k++) {
        std::fprintf(fp, "4 %d %d %d %d\n", 4 * k, 4 * k + 1, 4 * k + 3,
                     4 * k + 2);
      }

      // All quadrilaterals
      std::fprintf(fp, "\nCELL_TYPES %d\n", size);
      for (int k = 0; k < size; k++) {
        std::fprintf(fp, "%d\n", 9);
      }

      std::fclose(fp);
    }
  }

  /**
   * @brief Duplicate the quadrants
   *
   * @return std::shared_ptr<Quadtree>
   */
  std::shared_ptr<Quadtree> duplicate() const {
    auto tree = std::make_shared<Quadtree>();
    tree->quadrants = quadrants->duplicate();
    return tree;
  }

  /**
   * @brief Coarsen the quadrants in the quadtree
   *
   * @return std::shared_ptr<Quadtree>
   */
  std::shared_ptr<Quadtree> coarsen() const {
    QuadrantArray& quads = *quadrants;

    // Create a new queue of quadrants
    QuadrantHash hash;

    // Scan through the list, if we have offset quadrants which
    // all share the same parent, then coarsen the parent
    for (int i = 0; i < quads.size(); i++) {
      if (quads[i].level > 0) {
        if (quads[i].child_id() == 0) {
          Quadrant p = quads[i].parent();
          hash.add_quadrant(p);
        }
      } else {
        hash.add_quadrant(quads[i]);
      }
    }

    // Create the coarse quadrants
    auto coarse = std::make_shared<Quadtree>();
    coarse->quadrants = hash.to_array();

    return coarse;
  }

  /**
   * @brief Refine the mesh
   *
   * @param refinement The refinement array (must be of size > self->size())
   * @param min_level Minimum level
   * @param max_level Maximum level
   */
  void refine(const int refinement[] = nullptr, std::int32_t min_level = 0,
              std::int32_t max_level = Quadrant::MAX_LEVEL) {
    QuadrantArray& quads = *quadrants;

    // Adjust the min and max levels to ensure consistency
    if (min_level < 0) {
      min_level = 0;
    }
    if (max_level > Quadrant::MAX_LEVEL) {
      max_level = Quadrant::MAX_LEVEL;
    }

    // This is just a sanity check
    if (min_level > max_level) {
      min_level = max_level;
    }

    // Create a hash table for the refined quadrants and the quadrants
    // that are external (on other processors)
    QuadrantHash hash;

    if (refinement) {
      for (int i = 0; i < quads.size(); i++) {
        if (refinement[i] == 0) {
          // We know that this quadrant is locally owned
          hash.add_quadrant(quads[i]);
        } else if (refinement[i] < 0) {
          // Coarsen this quadrant
          if (quads[i].level > min_level) {
            // Compute the new refinement level
            int new_level = quads[i].level + refinement[i];
            if (new_level < min_level) {
              new_level = min_level;
            }

            // Copy over the quadrant
            Quadrant q = quads[i];
            q.level = new_level;
            q.info = 0;

            // Compute the new side-length of the quadrant
            const std::int32_t h = q.get_size();
            q.x = q.x - (q.x % h);
            q.y = q.y - (q.y % h);
            hash.add_quadrant(q);
          } else {
            // If it is already at the min level, just add it
            hash.add_quadrant(quads[i]);
          }
        } else if (refinement[i] > 0) {
          // Refine this quadrant
          if (quads[i].level < max_level) {
            // Compute the new refinement level
            int new_level = quads[i].level + refinement[i];
            if (new_level > max_level) {
              new_level = max_level;
            }

            // Compute the relative level of refinement
            int ref = new_level - quads[i].level;
            if (ref <= 0) {
              ref = 1;
            } else {
              ref = 1 << (ref - 1);
            }

            // Copy the quadrant and set the new level
            Quadrant q = quads[i];
            q.level = new_level;
            q.info = 0;

            // Compute the new side-length of the quadrant
            const std::int32_t h = q.get_size();
            std::int32_t x = q.x - (q.x % h);
            std::int32_t y = q.y - (q.y % h);
            for (int ii = 0; ii < ref; ii++) {
              for (int jj = 0; jj < ref; jj++) {
                q.x = x + 2 * ii * h;
                q.y = y + 2 * jj * h;
                hash.add_quadrant(q);
              }
            }
          } else {
            // If the quadrant is at the max level add it without
            // refinement
            hash.add_quadrant(quads[i]);
          }
        }
      }
    } else {
      // No refinement array is provided. Just go ahead and refine
      // everything one level
      for (int i = 0; i < quads.size(); i++) {
        if (quads[i].level < max_level) {
          Quadrant q = quads[i];
          q.level += 1;
          q.info = 0;
          hash.add_quadrant(q);
        } else {
          hash.add_quadrant(quads[i]);
        }
      }
    }

    quadrants = hash.to_array();
  }

  /**
   * @brief Balance the quadrants in the tree
   *
   * @param balance_corner Balance across corners too
   */
  void balance(bool balance_corner = true) {
    QuadrantArray& quads = *quadrants;

    // Create a hash table for the balanced tree
    QuadrantHash hash;
    QuadrantQueue queue;

    // Add all the elements
    for (int i = 0; i < quads.size(); i++) {
      Quadrant quad = quads[i].get_sibling(0);
      hash.add_quadrant(quad);

      // Balance the quadrants locally
      balance_quadrant(quad, hash, queue, balance_corner);
    }

    // Now continue until the queue of added quadrants is
    // empty. At each iteration, pop an quadrant and add
    // its neighbours until nothing new is added. This code
    // handles the propagation of quadrants to adjacent quadrants.
    while (!queue.empty()) {
      Quadrant quad = queue.pop();
      bool print_flag = true;
      balance_quadrant(quad, hash, queue, balance_corner, print_flag);
    }

    // Set the elements into the quadtree
    bool uniquify = false;
    std::shared_ptr<QuadrantArray> child0_array = hash.to_array(uniquify);
    QuadrantArray& child0 = *child0_array;

    // Add quadrants into the hashs
    for (int i = 0; i < child0.size(); i++) {
      if (child0[i].level > 0) {
        for (int j = 0; j < 4; j++) {
          Quadrant q = child0[i].get_sibling(j);
          hash.add_quadrant(q);
        }
      }
    }

    quadrants = hash.to_array();

    // Label the dependent edges for later usage
    label_dependent_edges();
  }

  /**
   * @brief Get the quadrants from the mesh
   *
   * @return std::shared_ptr<QuadrantArray>
   */
  std::shared_ptr<QuadrantArray> get_quadrants() { return quadrants; }

  /**
   * @brief Create a list of local nodes
   *
   * The level variable is used to store the number of nodes associated
   * with the group of nodes.
   *
   * @param degree Polynomial degree of the elements
   * @return std::shared_ptr<NodeArray> Array of nodes
   */
  std::shared_ptr<NodeArray> create_nodes(int degree) {
    // Allocate the array of elements
    QuadrantArray& quads = *quadrants;

    // Create all the nodes/edges/faces
    NodeHash hash;

    // First of all, add all the nodes from the local elements
    // on this processor
    for (int i = 0; i < quads.size(); i++) {
      for (int jj = 0; jj < degree + 1; jj++) {
        for (int ii = 0; ii < degree + 1; ii++) {
          QuadrantNode node = quads[i].get_node(degree, ii, jj);
          hash.add_node(node);
        }
      }
    }

    return hash.to_array();
  }

 private:
  // The array of quadrants
  std::shared_ptr<QuadrantArray> quadrants;

  void balance_quadrant(Quadrant& quad, QuadrantHash& hash,
                        QuadrantQueue& queue, const bool balance_corner,
                        bool print_flag = false) {
    // Get the max level
    const std::int32_t hmax = 1 << Quadrant::MAX_LEVEL;

    // Get the parent of the quadrant, and add the their
    // face-matched quadrants from each face, as long
    // as they fall within the bounds
    if (quad.level >= 1) {
      Quadrant p = quad.parent();

      // Add the edge-adjacent elements
      for (int edge = 0; edge < 4; edge++) {
        Quadrant neighbor = p.edge_neighbor(edge);
        Quadrant q = neighbor.get_sibling(0);

        // If we're in bounds, add the neighbor
        if ((q.x >= 0 && q.x < hmax) && (q.y >= 0 && q.y < hmax)) {
          if (hash.add_quadrant(q)) {
            queue.push(q);
          }
        }
      }

      // If we're balancing across edges and corners
      if (balance_corner) {
        for (int corner = 0; corner < 4; corner++) {
          Quadrant neighbor = p.corner_neighbor(corner);
          Quadrant q = neighbor.get_sibling(0);

          if ((q.x >= 0 && q.x < hmax) && (q.y >= 0 && q.y < hmax)) {
            if (hash.add_quadrant(q)) {
              queue.push(q);
            }
          }
        }
      }
    }
  }

  /**
   * @brief Label the dependent edges of the two quadrants adjacent to an edge
   * with hanging nodes
   */
  void label_dependent_edges() {
    QuadrantArray& quads = *quadrants;

    // Enumerate the sibling-ids for each edge
    const int edge_index_to_children[][2] = {{0, 2}, {1, 3}, {0, 1}, {2, 3}};
    const int edge_index_to_adjacent[] = {1, 0, 3, 2};

    // Max side length of the element
    const std::int32_t hmax = 1 << Quadrant::MAX_LEVEL;

    // Initialize the info
    for (int i = 0; i < quads.size(); i++) {
      quads[i].info = 0;
    }

    for (int i = 0; i < quads.size(); i++) {
      // Check whether the next-level refined element exists over an
      // adjacent edge
      for (int edge_index = 0; edge_index < 4; edge_index++) {
        for (int k = 0; k < 2; k++) {
          // Get the quadrant and increase the level. p is now one level more
          // refined than quads[i].
          Quadrant p = quads[i];
          p.level += 1;

          // Get the sibling id for each quadrant along the
          // face that we're on right now
          Quadrant q = p.get_sibling(edge_index_to_children[edge_index][k]);

          // Get the edge neighbor
          q = q.edge_neighbor(edge_index);

          // Find the edge and set the info flag
          if ((q.x >= 0 && q.x < hmax) && (q.y >= 0 && q.y < hmax)) {
            Quadrant* dep = quadrants->contains(q);
            if (dep) {
              // Set the info flag to the corresponding adjacent index
              dep->info |= 1 << edge_index_to_adjacent[edge_index];

              // Apply a label to the coarser source quadrant, indicating
              // the quadrant
              quads[i].info |= 1 << (4 + edge_index);
            }
          }
        }
      }
    }
  }
};

}  // namespace xcgd

#endif  // XCGD_QUADTREE_H