#include <glm/glm.hpp>
#include <vector>

const int MAX_TREE_DEPTH = 20;

struct Body {

	glm::vec3 position;
	glm::vec3 velocity;
	glm::vec3 force;
	glm::vec3 oldforce;
	float mass;

};

struct BoundingBox {

	glm::vec3 center;
	float halfWidth; // Center-Edge distance

	BoundingBox() { center = glm::vec3(0.0f); halfWidth = 0.0; }
	BoundingBox(glm::vec3 c, float hd) {
		center = c; halfWidth = hd;
	}

	bool contains(const glm::vec3& point) {
		return ((point.x >= center.x - halfWidth) && (point.x <= center.x + halfWidth) && 
			(point.y >= center.y - halfWidth) && (point.y <= center.y + halfWidth) && 
			(point.z >= center.z - halfWidth) && (point.z <= center.z + halfWidth));
	}

};

struct OctNode {

	BoundingBox bbox;
	int depth;

	glm::vec3 centerOfMass;
	float totalMass;

	std::vector<Body*> bodies;

	OctNode* children[8];
	bool isLeaf;

	OctNode(BoundingBox b, int d) {
		bbox = b;
		totalMass = 0;
		depth = d;;
		isLeaf = true;
		centerOfMass = glm::vec3(0.0f);
		for (int i = 0; i < 8; i++) {
			children[i] = nullptr;
		}
	}

	~OctNode(){
		for (int i = 0; i < 8; i++) {
			if (!(children[i] == nullptr)) {
				delete children[i];
				children[i] = nullptr;
			}
		}
	}

	void insert(Body* newbody) {

		if (depth >= MAX_TREE_DEPTH) {
			bodies.push_back(newbody);
		}

		if (this->isLeaf) {  // Leaf Node

			if (this->bodies.empty()) { // Empty

				this->bodies.push_back(newbody);
				return;

			}
			else { // Not empty

				subdivide();
				int oldOctant = getOctant(this->bodies[0]->position);
				children[oldOctant]->insert(this->bodies[0]);

				this->bodies.clear();
				this->isLeaf = false;

			}
		}
		
		
		int newOctant = getOctant(newbody->position);
		children[newOctant]->insert(newbody);

	}
	void computeMassDistribution() {

		// Leaf node
		if (isLeaf) {
			if (!bodies.empty()) {

				totalMass = 0.0f;
				centerOfMass = glm::vec3(0.0f);

				for (Body* b : bodies) {
					totalMass += b->mass;
					centerOfMass += b->position;
				}
				if (totalMass > 0.0f) centerOfMass /= totalMass;

			}
			else {

				centerOfMass = glm::vec3(0.0f);
				totalMass = 0;

			}
			return;
		}

		// Recursive call for internal node

		totalMass = 0.0f;
		centerOfMass = glm::vec3(0.0f);

		for (int i = 0; i < 8; i++) {
			if (children[i] != nullptr) {
				children[i]->computeMassDistribution();

				totalMass += children[i]->totalMass;
				centerOfMass += children[i]->centerOfMass * children[i]->totalMass;
			}
		}

		if (totalMass > 0.0f) {
			centerOfMass /= totalMass;
		}

	}

private:

	void subdivide() {

		float quarterWidth = bbox.halfWidth / 2;

		for (int i = 0; i < 8; i++) {
			glm::vec3 newCenter = bbox.center;

			newCenter.x += (i & 1) ? quarterWidth : -quarterWidth;
			newCenter.y += (i & 2) ? quarterWidth : -quarterWidth;
			newCenter.z += (i & 4) ? quarterWidth : -quarterWidth;

			children[i] = new OctNode(BoundingBox(newCenter, quarterWidth), depth+1);
		}

	}

	int getOctant(glm::vec3 position) {

		int octant = 0;
		if (position.x >= bbox.center.x) octant |= 1; // Bit 1
		if (position.y >= bbox.center.y) octant |= 2; // Bit 2
		if (position.z >= bbox.center.z) octant |= 4; // Bit 3
		return octant;

	}

};