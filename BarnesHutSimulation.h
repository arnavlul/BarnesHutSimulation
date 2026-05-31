#include <glm/glm.hpp>
#include <vector>

const int MAX_TREE_DEPTH = 20;

struct Body {

	glm::vec3 position;
	glm::vec3 velocity;
	glm::vec3 acceleration;
	glm::vec3 jerk = glm::vec3(0.0f);

	glm::vec3 predposition = glm::vec3(0.0f);
	glm::vec3 predvelocity = glm::vec3(0.0f);

	glm::vec3 oldacceleration = glm::vec3(0.0f);
	glm::vec3 oldjerk = glm::vec3(0.0f);

	float t_current = 0.0f;
	float block_dt = 0.0f;

	float mass;
	float radius;
	int submarineID = -1;
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

struct Submarine {

	int id;
	std::vector<int> particleIndices;
	glm::vec3 COM;
	glm::vec3 velCOM;
	float totalMass;
	float boundingRadius;

};

class SubmarineManager {
public:

	float linkingLength = 2.0f;
	unsigned int min_particles = 3;

	void findSubmarine(std::vector<Body>& bodies, std::vector<Submarine>& submarines) {

		submarines.clear();
		int nextID = 0;

		for (Body& b : bodies) b.submarineID = -1;

		for (int i = 0; i < bodies.size(); i++) {

			if (bodies[i].submarineID != -1) continue;

			std::vector<int> cluster;
			std::vector<int> queue;

			queue.push_back(i);
			bodies[i].submarineID = -2;

			int head = 0;
			while (head < queue.size()) {

				int currentIdx = queue[head++];
				cluster.push_back(currentIdx);

				for (int j = 0; j < bodies.size(); j++) {
					if (bodies[j].submarineID != -1) continue;

					float dist = glm::distance(bodies[j].position, bodies[currentIdx].position);
					if (dist < linkingLength) {
						bodies[j].submarineID = -2;
						queue.push_back(j);
					}
				}
			}

			if (cluster.size() >= min_particles) {
				Submarine sub;
				sub.id = nextID;
				sub.particleIndices = cluster;

				sub.totalMass = 0.0f;
				sub.COM = glm::vec3(0.0f);
				sub.velCOM = glm::vec3(0.0f);

				for (int i : cluster) {
					bodies[i].submarineID = nextID;
					sub.totalMass += bodies[i].mass;
					sub.COM += bodies[i].position * bodies[i].mass;
					sub.velCOM += bodies[i].velocity * bodies[i].mass;
				}
				sub.COM /= sub.totalMass;
				sub.velCOM /= sub.totalMass;

				float maxDistSq = 0.0f;
				for (int i : cluster) {
					glm::vec3 dir = bodies[i].position - sub.COM;
					float dist2 = dir.x * dir.x + dir.y * dir.y + dir.z * dir.z;
					if (dist2 > maxDistSq) maxDistSq = dist2;
				}
				sub.boundingRadius = sqrt(maxDistSq);
				submarines.push_back(sub);
				nextID++;
			}
			else {
				for (int i : cluster) bodies[i].submarineID = -1;
			}

		}

	}

};