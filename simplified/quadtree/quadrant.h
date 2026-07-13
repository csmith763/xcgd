#ifndef XCGD_QUADRANT_H
#define XCGD_QUADRANT_H

#include <cstdint>
#include <queue>
#include <set>
#include <vector>

namespace xcgd {

class Quadrant {
 public:
  static constexpr std::int32_t MAX_LEVEL = 30;

  static constexpr std::int16_t NODE_LABEL = 1;
  static constexpr std::int16_t EDGE_LABEL = 2;
  static constexpr std::int16_t FACE_LABEL = 4;

  int child_id() const {
    int id = 0;
    const std::int32_t h = 1 << (MAX_LEVEL - level);

    id = id | ((x & h) ? 1 : 0);
    id = id | ((y & h) ? 2 : 0);

    return id;
  }
  Quadrant get_sibling(int id) const {
    const std::int32_t h = 1 << (MAX_LEVEL - level);

    std::int32_t xr = ((x & h) ? x - h : x);
    std::int32_t yr = ((y & h) ? y - h : y);

    Quadrant sib;
    sib.level = level;
    sib.info = 0;
    sib.x = ((id & 1) ? xr + h : xr);
    sib.y = ((id & 2) ? yr + h : yr);

    return sib;
  }
  Quadrant parent() const {
    Quadrant p;
    if (level > 0) {
      p.level = level - 1;
      p.info = 0;
      const int32_t h = 1 << (MAX_LEVEL - level);

      p.x = x & ~h;
      p.y = y & ~h;
    } else {
      p.level = 0;
      p.info = 0;
      p.x = x;
      p.y = y;
    }
    return p;
  }
  Quadrant edge_neighbor(int edge) const {
    Quadrant neighbor;

    const std::int32_t h = 1 << (MAX_LEVEL - level);
    neighbor.level = level;
    neighbor.info = 0;

    neighbor.x = x + ((edge == 0) ? -h : (edge == 1) ? h : 0);
    neighbor.y = y + ((edge == 2) ? -h : (edge == 3) ? h : 0);
    return neighbor;
  }
  Quadrant corner_neighbor(int corner) const {
    Quadrant neighbor;

    const std::int32_t h = 1 << (MAX_LEVEL - level);
    neighbor.level = level;
    neighbor.info = 0;

    neighbor.x = x + (2 * (corner & 1) - 1) * h;
    neighbor.y = y + ((corner & 2) - 1) * h;

    return neighbor;
  }
  bool contains(const Quadrant& quad) const {
    const std::int32_t h = 1 << (MAX_LEVEL - level);

    // Check whether the quadrant lies within this quadrant
    if ((quad.x >= x && quad.x < x + h) && (quad.y >= y && quad.y < y + h)) {
      return true;
    }

    return false;
  }

  static int compare(const Quadrant& a, const Quadrant& b) {
    std::uint32_t xxor = a.x ^ b.x;
    std::uint32_t yxor = a.y ^ b.y;
    std::uint32_t sor = xxor | yxor;

    // If there is no most-significant bit, then we are done
    if (sor == 0) {
      return a.level - b.level;
    }

    // Check for the most-significant bit
    int discrim = 0;
    if (xxor > (sor ^ xxor)) {
      discrim = a.x - b.x;
    } else {
      discrim = a.y - b.y;
    }

    if (discrim > 0) {
      return 1;
    } else if (discrim < 0) {
      return -1;
    }

    return 0;
  }

  static int compare_position(const Quadrant& a, const Quadrant& b) {
    std::uint32_t xxor = a.x ^ b.x;
    std::uint32_t yxor = a.y ^ b.y;
    std::uint32_t sor = xxor | yxor;

    // Note that here we do not distinguish between levels
    // Check for the most-significant bit
    int discrim = 0;
    if (xxor > (sor ^ xxor)) {
      discrim = a.x - b.x;
    } else {
      discrim = a.y - b.y;
    }

    if (discrim > 0) {
      return 1;
    } else if (discrim < 0) {
      return -1;
    }

    return 0;
  }

  std::int32_t x, y;   // The x,y coordinates
  std::int32_t tag;    // A tag to store additional data
  std::int16_t level;  // The refinement level
  std::int16_t info;   // The info about faces
};

/**
 * @brief A fixed-size array of unique quadrant objects.
 *
 * During construction, the quadrants are sorted and unquified so that the array
 * does not contain duplicates and are in a Morton ordering
 */
class QuadrantArray {
 public:
  QuadrantArray(std::vector<Quadrant> vec, bool uniquify = true)
      : quads(std::move(vec)) {
    sort_and_uniquify(uniquify);
  }

  int size() const { return static_cast<int>(quads.size()); }
  Quadrant& operator[](int i) { return quads[i]; }
  const Quadrant& operator[](int i) const { return quads[i]; }

  Quadrant* contains(const Quadrant& q) {
    auto it = std::lower_bound(quads.begin(), quads.end(), q,
                               [](const Quadrant& a, const Quadrant& b) {
                                 return Quadrant::compare(a, b) < 0;
                               });

    if (it != quads.end() && Quadrant::compare(*it, q) == 0) {
      return &(*it);
    }

    return nullptr;
  }

  std::shared_ptr<QuadrantArray> duplicate() const {
    return std::make_shared<QuadrantArray>(quads);
  }

 private:
  void sort_and_uniquify(bool uniquify = true) {
    std::sort(quads.begin(), quads.end(),
              [](const Quadrant& a, const Quadrant& b) {
                return Quadrant::compare(a, b) < 0;
              });

    if (uniquify) {
      // Now that the Quadrants are sorted, remove duplicates
      int i = 0;  // Location from which to take entries
      int j = 0;  // Location to place entries
      int size = quads.size();

      for (; i < size; i++, j++) {
        while ((i < size - 1) &&
               (Quadrant::compare_position(quads[i], quads[i + 1]) == 0)) {
          i++;
        }

        if (i != j) {
          quads[j] = quads[i];
        }
      }

      // The new size of the array
      size = j;

      quads.erase(quads.begin() + size, quads.end());
    }
  }

  std::vector<Quadrant> quads;
};

/**
 * @brief A queue of quadrants.
 *
 */
class QuadrantQueue {
 public:
  QuadrantQueue() {}

  /**
   * @brief Clear the queue
   */
  void clear() { queue = {}; }

  /**
   * @brief Return the size of the queue
   *
   * @return int size of the queue
   */
  int size() const { return queue.size(); }

  /**
   * @brief Check if the queue is empty
   */
  bool empty() const { return queue.empty(); }

  /**
   * @brief Push a quadrant into the queue
   *
   * @param quad The quadrant
   */
  void push(const Quadrant& quad) { queue.push(quad); }

  /**
   * @brief Pop a quadrant from the queue
   *
   * @return Quadrant The resulting quadrant
   */
  Quadrant pop() {
    Quadrant q = queue.front();
    queue.pop();
    return q;
  }

  /**
   * @brief Convert the quadrants into an array
   *
   * @return std::shared_ptr<QuadrantArray>
   */
  std::shared_ptr<QuadrantArray> to_array(bool uniquify = true) const {
    std::vector<Quadrant> quads;
    quads.reserve(queue.size());

    std::queue<Quadrant> copy = queue;

    while (!copy.empty()) {
      quads.push_back(copy.front());
      copy.pop();
    }

    return std::make_shared<QuadrantArray>(quads, uniquify);
  }

 private:
  std::queue<Quadrant> queue;
};

/**
 * @brief A hash object to uniquely store quadrants
 */
class QuadrantHash {
 public:
  QuadrantHash() {}

  /**
   * @brief Clear all the stored quadrants in the hash
   */
  void clear() { quads.clear(); }

  /**
   * @brief Try to add a quadrant to the hash
   *
   * @param quad The quadrant to attempt to add
   * @return true The quadrant was added
   * @return false The quadrant already exists
   */
  bool add_quadrant(const Quadrant& quad) {
    auto result = quads.insert(quad);
    return result.second;
  }

  /**
   * @brief Convert the unique quadrants into a unique array
   *
   * @return std::shared_ptr<QuadrantArray>
   */
  std::shared_ptr<QuadrantArray> to_array(bool uniquify = true) const {
    std::vector<Quadrant> vec;
    vec.reserve(quads.size());

    for (const Quadrant& q : quads) {
      vec.push_back(q);
    }

    return std::make_shared<QuadrantArray>(std::move(vec), uniquify);
  }

 private:
  struct QuadrantLess {
    bool operator()(const Quadrant& a, const Quadrant& b) const {
      return Quadrant::compare(a, b) < 0;
    }
  };

  std::set<Quadrant, QuadrantLess> quads;
};

/**
 * @brief A fixed-size array of unique nodes.
 *
 * The same Quadrant objects store the node, but now the level doesn't matter.
 * Only the x/y location of the quadrant.
 *
 * During construction, the quadrants are sorted and unquified so that the array
 * does not contain duplicates and are in a Morton ordering
 */
class NodeArray {
 public:
  NodeArray(std::vector<Quadrant> vec) : quads(std::move(vec)) {
    sort_and_uniquify();
  }

  int size() const { return static_cast<int>(quads.size()); }
  Quadrant& operator[](int i) { return quads[i]; }
  const Quadrant& operator[](int i) const { return quads[i]; }

  Quadrant* contains(const Quadrant& q) {
    auto it = std::lower_bound(quads.begin(), quads.end(), q,
                               [](const Quadrant& a, const Quadrant& b) {
                                 return Quadrant::compare_position(a, b) < 0;
                               });

    if (it != quads.end() && Quadrant::compare_position(*it, q) == 0) {
      return &(*it);
    }

    return nullptr;
  }

  int get_index(const Quadrant& q) const {
    auto it = std::lower_bound(quads.begin(), quads.end(), q,
                               [](const Quadrant& a, const Quadrant& b) {
                                 return Quadrant::compare_position(a, b) < 0;
                               });

    if (it != quads.end() && Quadrant::compare_position(*it, q) == 0) {
      return it - quads.begin();
    }

    return -1;
  }

  std::shared_ptr<NodeArray> duplicate() const {
    return std::make_shared<NodeArray>(quads);
  }

 private:
  void sort_and_uniquify(bool uniquify = true) {
    std::sort(quads.begin(), quads.end(),
              [](const Quadrant& a, const Quadrant& b) {
                return Quadrant::compare_position(a, b) < 0;
              });

    auto last = std::unique(quads.begin(), quads.end(),
                            [](const Quadrant& a, const Quadrant& b) {
                              return Quadrant::compare_position(a, b) == 0;
                            });

    quads.erase(last, quads.end());
  }

  std::vector<Quadrant> quads;
};

/**
 * @brief A hash object to uniquely store nodes
 */
class NodeHash {
 public:
  NodeHash() {}

  /**
   * @brief Clear all the stored nodes in the hash
   */
  void clear() { quads.clear(); }

  /**
   * @brief Try to add a node to the hash
   *
   * @param quad The quadrant to attempt to add
   * @return true The quadrant was added
   * @return false The quadrant already exists
   */
  bool add_node(const Quadrant& quad) {
    auto result = quads.insert(quad);
    return result.second;
  }

  /**
   * @brief Convert the unique nodes into an array
   *
   * @return std::shared_ptr<NodeArray>
   */
  std::shared_ptr<NodeArray> to_array() const {
    std::vector<Quadrant> vec;
    vec.reserve(quads.size());

    for (const Quadrant& q : quads) {
      vec.push_back(q);
    }

    return std::make_shared<NodeArray>(std::move(vec));
  }

 private:
  struct NodeLess {
    bool operator()(const Quadrant& a, const Quadrant& b) const {
      return Quadrant::compare_position(a, b) < 0;
    }
  };

  std::set<Quadrant, NodeLess> quads;
};

}  // namespace xcgd

#endif  // XCGD_QUADRANT_H