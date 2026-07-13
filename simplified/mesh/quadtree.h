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
        const std::int32_t h = 1 << (Quadrant::MAX_LEVEL - quads[k].level);

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
            const std::int32_t h = 1 << (Quadrant::MAX_LEVEL - q.level);
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
            const std::int32_t h = 1 << (Quadrant::MAX_LEVEL - q.level);
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
   * @brief Create the connectivity for the given quadtree
   *
   * @param degree Degree of the mesh to create
   * @return
   */
  std::vector<int> create_connectivity(int degree = 1) {
    if (degree < 1) {
      degree = 1;
    }

    // Create the nodes
    std::shared_ptr<NodeArray> node_array = create_nodes(degree);
    NodeArray& nodes = *node_array;

    // Create offsets into the array of nodes
    std::vector<int> node_offsets(nodes.size());

    for (int i = 0, local_size = 0; i < nodes.size(); i++) {
      node_offsets[i] = local_size;
      local_size += nodes[i].level;
    }

    // Allocate the connectivity
    std::vector<int> conn;
    compute_connectivity(degree, nodes, node_offsets, conn);

    return conn;
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
          // Get the quadrant and increase the level
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
            }
          }
        }
      }
    }
  }

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
    bool use_node_index = true;
    NodeHash hash;

    // Set the node, edge and face label
    std::int16_t node_label = 0, edge_label = 0, face_label = 0;
    if (degree <= 2) {
      node_label = Quadrant::NODE_LABEL;
      edge_label = Quadrant::NODE_LABEL;
      face_label = Quadrant::NODE_LABEL;
    } else {
      node_label = Quadrant::NODE_LABEL;
      edge_label = Quadrant::EDGE_LABEL;
      face_label = Quadrant::FACE_LABEL;
    }

    // Set the node locations
    if (degree == 1) {
      // First of all, add all the nodes from the local elements
      // on this processor
      for (int i = 0; i < quads.size(); i++) {
        const int32_t h = 1 << (Quadrant::MAX_LEVEL - quads[i].level);
        for (int jj = 0; jj < 2; jj++) {
          for (int ii = 0; ii < 2; ii++) {
            Quadrant node;
            node.level = 1;
            node.x = quads[i].x + h * ii;
            node.y = quads[i].y + h * jj;
            node.tag = 0;
            node.info = node_label;
            hash.add_node(node);
          }
        }
      }

      // Add the nodes that the dependent nodes depend on
      for (int i = 0; i < quads.size(); i++) {
        // Add the external nodes from dependent edges
        if (quads[i].info) {
          for (int edge_index = 0; edge_index < 4; edge_index++) {
            if (quads[i].info & 1 << edge_index) {
              Quadrant parent = quads[i].parent();

              const int32_t hp = 1 << (Quadrant::MAX_LEVEL - parent.level);
              for (int ii = 0; ii < 2; ii++) {
                Quadrant node;
                node.level = 1;
                if (edge_index < 2) {
                  node.x = parent.x + hp * (edge_index % 2);
                  node.y = parent.y + hp * ii;
                } else {
                  node.x = parent.x + hp * ii;
                  node.y = parent.y + hp * (edge_index % 2);
                }
                // Assign a negative rank index for now...
                node.tag = -1;
                node.info = node_label;
                hash.add_node(node);
              }
            }
          }
        }
      }
    } else {
      for (int i = 0; i < quads.size(); i++) {
        const int32_t h = 1 << (Quadrant::MAX_LEVEL - quads[i].level - 1);
        for (int jj = 0; jj < 3; jj++) {
          for (int ii = 0; ii < 3; ii++) {
            Quadrant node;
            node.x = quads[i].x + h * ii;
            node.y = quads[i].y + h * jj;
            if ((ii == 0 || ii == 2) && (jj == 0 || jj == 2)) {
              node.level = 1;
              node.info = node_label;
            } else if (ii == 0 || ii == 2 || jj == 0 || jj == 2) {
              node.level = degree - 1;
              node.info = edge_label;
            } else {
              node.level = (degree - 1) * (degree - 1);
              node.info = face_label;
            }
            node.tag = 0;
            hash.add_node(node);
          }
        }
      }

      // Add the nodes from the dependent edges
      for (int i = 0; i < quads.size(); i++) {
        if (quads[i].info) {
          const int32_t h = 1 << (Quadrant::MAX_LEVEL - quads[i].level);

          for (int edge_index = 0; edge_index < 4; edge_index++) {
            if (quads[i].info & 1 << edge_index) {
              Quadrant parent = quads[i].parent();

              for (int ii = 0; ii < 3; ii++) {
                Quadrant node;

                node.level = 0;

                // Set the location of the edge
                if (edge_index < 2) {
                  node.x = parent.x + 2 * h * (edge_index % 2);
                  node.y = parent.y + h * ii;
                } else {
                  node.x = parent.x + h * ii;
                  node.y = parent.y + 2 * h * (edge_index % 2);
                }

                if (ii == 0 || ii == 2) {
                  node.level = 1;
                  node.info = node_label;
                } else {
                  // For a dependent-edge which connects to another element with
                  // fewer nodes, the number of independent dof drops because
                  // some nodes are shared, but not all
                  node.level = degree - 2;
                  node.info = edge_label;
                }
                // Assign the negative rank to this processor
                node.tag = 0;
                hash.add_node(node);
              }
            }
          }
        }
      }
    }

    return hash.to_array();
  }

  /**
   * @brief Build the connectivity arrays for each element
   *
   * @param degree Element degree
   * @param nodes Node array associated with each node group
   * @param node_offsets Offset into the node
   * @param conn Generated connectivity array
   */
  void compute_connectivity(int degree, const NodeArray& nodes,
                            const std::vector<int>& node_offsets,
                            std::vector<int>& conn) {
    QuadrantArray& quads = *quadrants;

    // Set the node, edge and face label
    std::int16_t node_label = 0, edge_label = 0, face_label = 0;
    if (degree <= 2) {
      node_label = Quadrant::NODE_LABEL;
      edge_label = Quadrant::NODE_LABEL;
      face_label = Quadrant::NODE_LABEL;
    } else {
      node_label = Quadrant::NODE_LABEL;
      edge_label = Quadrant::EDGE_LABEL;
      face_label = Quadrant::FACE_LABEL;
    }

    // Allocate the connectivity
    std::size_t size = (degree + 1) * (degree + 1) * quads.size();
    conn.resize(size);

    if (degree <= 2) {
      for (int i = 0; i < quads.size(); i++) {
        int* c = &conn[(degree + 1) * (degree + 1) * i];
        const int32_t h = 1 << (Quadrant::MAX_LEVEL - quads[i].level - 1);

        // Loop over the element nodes
        for (int corner_index = 0; corner_index < 4; corner_index++) {
          Quadrant node;
          node.x = quads[i].x + 2 * h * (corner_index % 2);
          node.y = quads[i].y + 2 * h * (corner_index / 2);
          node.info = node_label;

          int index = nodes.get_index(node);
          int offset = degree * (corner_index % 2) +
                       degree * (degree + 1) * (corner_index / 2);
          c[offset] = node_offsets[index];
        }

        if (degree == 2) {
          // Loop over the edges and get the owners
          for (int edge_index = 0; edge_index < 4; edge_index++) {
            Quadrant node;
            if (edge_index < 2) {
              node.x = quads[i].x + 2 * h * (edge_index % 2);
              node.y = quads[i].y + h;
            } else {
              node.x = quads[i].x + h;
              node.y = quads[i].y + 2 * h * (edge_index % 2);
            }
            node.info = edge_label;

            int index = nodes.get_index(node);
            if (edge_index < 2) {
              int offset = (degree + 1) + degree * edge_index;
              c[offset] = node_offsets[index];
            } else {
              int offset = 1 + degree * (degree + 1) * (edge_index % 2);
              c[offset] = node_offsets[index];
            }
          }

          Quadrant node;
          node.x = quads[i].x + h;
          node.y = quads[i].y + h;
          node.info = face_label;
          int index = nodes.get_index(node);
          c[4] = node_offsets[index];
        }
      }
    } else {
      // Loop over all the elements and assign the local index owners
      // for each node
      for (int i = 0; i < quads.size(); i++) {
        int* c = &conn[(degree + 1) * (degree + 1) * i];

        // Compute the half-edge length of the quadrant
        const int32_t h = 1 << (Quadrant::MAX_LEVEL - quads[i].level - 1);

        // Loop over the corner nodes
        for (int corner_index = 0; corner_index < 4; corner_index++) {
          // Compute the offset to the local node
          int offset = degree * (corner_index % 2) +
                       degree * (degree + 1) * (corner_index / 2);

          // Find the node at the corner to determine the owner
          Quadrant node;
          node.x = quads[i].x + 2 * h * (corner_index % 2);
          node.y = quads[i].y + 2 * h * (corner_index / 2);
          node.info = node_label;
          int index = nodes.get_index(node);
          c[offset] = node_offsets[index];
        }

        // Loop over the edges and get the owners
        for (int edge_index = 0; edge_index < 4; edge_index++) {
          Quadrant edge;
          if (edge_index < 2) {
            edge.x = quads[i].x + 2 * h * (edge_index % 2);
            edge.y = quads[i].y + h;
          } else {
            edge.y = quads[i].y + 2 * h * (edge_index % 2);
            edge.x = quads[i].x + h;
          }
          edge.info = edge_label;
          int index = nodes.get_index(edge);

          if (edge_index < 2) {
            for (int k = 1; k < degree; k++) {
              int offset = k * (degree + 1) + degree * edge_index;
              c[offset] = node_offsets[index] + k - 1;
            }
          } else {
            for (int k = 1; k < degree; k++) {
              int offset = k + degree * (degree + 1) * (edge_index % 2);
              c[offset] = node_offsets[index] + k - 1;
            }
          }
        }

        // Loop over the face owners
        Quadrant face;
        face.x = quads[i].x + h;
        face.y = quads[i].y + h;
        face.info = face_label;
        int index = nodes.get_index(face);

        for (int jj = 1; jj < degree; jj++) {
          for (int ii = 1; ii < degree; ii++) {
            int offset = ii + jj * (degree + 1);
            c[offset] =
                node_offsets[index] + (ii - 1) + (jj - 1) * (degree - 1);
          }
        }
      }
    }
  }
};

}  // namespace xcgd

#endif  // XCGD_QUADTREE_H