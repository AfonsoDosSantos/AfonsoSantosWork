#include "../include/rrt_solver.h"
#include <rclcpp/rclcpp.hpp>
#include <memory>
#include <tuple>
#include <vector>
#include <bits/stdc++.h> 
#include <iostream>
#include <urdf/model.h>
#include "chain.hpp"
#include "tree.hpp"
#include "kdl.hpp"
#include <kdl_parser/kdl_parser.hpp>
#include "frames.hpp"
#include "jntarray.hpp"
#include <cmath>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <chrono>
#include <algorithm>
#include <execution>
#include <fstream>
#include <yaml-cpp/yaml.h>
#include <rclcpp/node.hpp>
#include <rclcpp/node_interfaces/node_base_interface.hpp>
#include <rclcpp/logging.hpp>
#include <urdf_parser/urdf_parser.h>
#include "../lib/orocos_kdl/install/include/kdl/chainfksolverpos_recursive.hpp"


namespace RRT{
    /**
     * Unzips a vector of pairs into two separate vectors.
     * 
     * @param zipped: The vector of pairs to be unzipped.
     * @param a: The vector to store the first elements of the pairs.
     * @param b: The vector to store the second elements of the pairs.
     */
    template <typename A, typename B>
    void unzip(const std::vector<std::pair<A, B>> &zipped, std::vector<A> &a, std::vector<B> &b){
        for(size_t i=0; i<a.size(); i++)
        {
            a[i] = zipped[i].first;
            b[i] = zipped[i].second;
        }
    }


    /**
     * Checks if the current sphere collides with another sphere.
     * 
     * @param sphere: The sphere to check for collision.
     * @return: True if collision occurs, false otherwise.
     */
    bool Sphere::collides_with_sphere(Sphere& sphere)
    {
        auto distance = (sphere.frame.p - frame.p).Norm();
        if (distance <= sphere.r + r)
            return true;

        return false;
    }


    /**
     * Destructor for the RobotHull class.
     * Deletes all the dynamically allocated parts and clears the parts vector.
     */
    RobotHull::~RobotHull(){
        for (auto part : parts){
            delete part;
        }
        parts.clear();
    }


    /**
     * Updates the poses of all parts in the robot hull based on the provided 
     * joint values.
     * 
     * @param q: The joint values to update the robot hull.
     * @return: The vector of ICollidable pointers representing the updated parts.
     */
    std::vector<ICollidable*> RobotHull::collidables(Eigen::Matrix<double, 6, 1> q, std::shared_ptr<RobotHull> rob){
        for (size_t i = 0; i < parts.size(); i++){
            joint_arrs[i]->data = q.head(joint_arrs[i]->rows());
            auto chain = rob->chains[i];
            fk_solvers[i].JntToCart(*joint_arrs[i], frame_tmp, chain.getNrOfSegments(), chain);
            parts[i]->update_pose(frame_tmp);
        }
        return parts;
    }


    /**
     * Adds a part to the robot hull.
     * 
     * @param fk_solver: The forward kinematics solver for the part.
     * @param obj: The ICollidable object representing the part.
     * @param nm_joints: The number of joints for the part.
     */
    void RobotHull::add_part(const KDL::ChainFkSolverPos_recursive& fk_solver, ICollidable* obj, unsigned int nm_joints, KDL::Chain& chain){
        fk_solvers.push_back(fk_solver);
        chains.push_back(chain);
        parts.push_back(obj);
        auto jnt = std::make_unique<KDL::JntArray>(nm_joints);
        joint_arrs.push_back(std::move(jnt));
    }


    /**
     * Constructor for the Node class.
     * 
     * @param q: The joint values of the node.
     * @param parent: A pointer to the parent node.
     */
    Node::Node(Eigen::Matrix<double, 6, 1> q, Node* parent){
        q_ = q;
        parent_ = parent;
    }


    /**
     * Adds a child node to the current node.
     * 
     * @param child: Pointer to the child node to be added.
     */
    void Node::add_child(Node* child){
        children.push_back(child);
    }


    /**
     * Returns the number of child nodes of the current node.
     * 
     * @return: The number of child nodes.
     */
    size_t Node::child_count() const{
        return children.size();
    }


    /**
     * Checks if the current node is the same as another node.
     * 
     * @param n: The node to compare with.
     * @return: True if the nodes are the same, false otherwise.
     */
    bool Node::same_node(const Node& n){
        if (&n == this)
            return true;
        else
            return false;
    }


    /**
     * Constructor for the Tree class.
     * @param q: The joint values of the start node.
     */
    Tree::Tree(Eigen::Matrix<double, 6, 1> q){
        start = new Node(q, nullptr);
        nodes.push_back(start);
    }

    
    /**
     * Destructor for the Tree class.
     * Deletes all the dynamically allocated nodes and clears the nodes vector.
     */
    Tree::~Tree(){
        for (auto p : nodes){
            delete p;
        }
        nodes.clear();
    }


    /**
     * Finds the nearest node in the tree to a given joint configuration.
     * 
     * @param q: The joint configuration to find the nearest node to.
     * @return: Pointer to the nearest node.
     */
    Node* Tree::nearest_node(Eigen::Matrix<double, 6, 1> q){
        double min_dist;
        Node* nearest;

        for (unsigned int i = 0; i < nodes.size(); i++){
            double res = configuration_distance(nodes[i]->q_, q);
            if (res < min_dist || i == 0){
                min_dist = res;
                nearest = nodes[i];
            }
        }
        return nearest;
    }


    /**
     * Adds a new node to the tree with a given joint configuration and a parent
     * node.
     * 
     * @param parent: Pointer to the parent node.
     * @param q: The joint configuration of the new node.
     */
    void Tree::add_node(Node* parent, Eigen::Matrix<double, 6, 1> q){
        auto newnode = new Node(q, parent);
        nodes.push_back(newnode);
        parent->add_child(newnode);
    }


    /**
     * Merges the current tree with another tree if their nodes are within a
     * specified limit distance.
     * 
     * @param tree: Pointer to the other tree to merge.
     * @param limit: The maximum distance for nodes to be considered for merging.
     * @return: True if a merge is possible, false otherwise.
     */
    bool Tree::merge(Tree* tree, double limit){
        std::vector<Node*>::iterator i;
        std::vector<Node*>::iterator i_other;

        for (i = nodes.begin();  i < nodes.end(); ++i){
            for (i_other = tree->nodes.begin();  i_other < tree->nodes.end(); ++i_other){
                double distance = configuration_distance((*i)->q_, (*i_other)->q_);
                if (distance <= limit){
                    // Merge possible
                    apply_merge((*i), (*i_other));
                    return true;
                } 
            }
        }
        return false;
    }


    /**
     * Applies a merge operation between two nodes.
     * 
     * @param parent_tail: Pointer to the parent node of the merge operation.
     * @param child_tail: Pointer to the child node of the merge operation.
     */
    void Tree::apply_merge(Node* parent_tail, Node* child_tail){
        Node* node_to_copy = child_tail;
        Node* parent_node = parent_tail;

        while (node_to_copy != nullptr){
            add_node(parent_node, node_to_copy->q_);
            parent_node = nodes.back();
            node_to_copy = node_to_copy->parent_;
        }
        end = nodes.back();
    }


    /**
     * Finds the closest pairs of nodes between the current tree and another tree.
     * 
     * @param tree: Pointer to the other tree.
     * @return: Vector of pairs of closest nodes between the trees.
     */
    std::vector<std::pair<Node*, Node*>> Tree::closest_nodepairs_of_trees(Tree* tree){
        using std::vector;
        using std::pair;

        vector<Node*>::iterator i;
        vector<Node*>::iterator i_other;

        vector<double> distances;
        distances.reserve(nodes.size() * tree->nodes.size());
        vector<pair<Node*, Node*>> nodepairs;
        nodepairs.reserve(nodes.size() * tree->nodes.size());

        // Calculate distances between all node pairs
        for (i = nodes.begin();  i < nodes.end(); ++i){
            for (i_other = tree->nodes.begin();  i_other < tree->nodes.end(); ++i_other){
                double distance = configuration_distance((*i)->q_, (*i_other)->q_);
                distances.push_back(distance);
                nodepairs.push_back(std::make_pair(*i, *i_other));
            }
        }
        // Sort the vector of pairs based on distances
        vector<pair<double, pair<Node*, Node*>>> zipped;
        zip(distances, nodepairs, zipped);
        
        std::sort(std::begin(zipped),
                  std::end(zipped),
                  [&](const auto& a,
                  const auto& b){
            return a.first < b.first;
        });

        unzip(zipped, distances, nodepairs);

        return nodepairs;
    }


    /**
     * Returns the path from the start node to the end node in the tree.
     * 
     * @return: Vector of nodes representing the path.
     */
    std::vector<Node*> Tree::path(){
        std::vector<Node*> pth;
        
        if (end != nullptr){
            Node* parent = end->parent_;
            pth.push_back(end);
            while (parent != nullptr){
                pth.push_back(parent);
                parent = parent->parent_;
            }
            std::reverse(pth.begin(), pth.end());
        }
        return pth;
    }


    /**
     * Retrieves the forward kinematics solver for a given chain in a KDL tree.
     * 
     * @param root_link: Name of the root link of the chain.
     * @param end_link: Name of the end link of the chain.
     * @param tree: KDL tree containing the chain.
     * @return: Chain forward kinematics solver.
     */
    KDL::ChainFkSolverPos_recursive get_fk_solver(std::string& root_link, 
                                                  std::string& end_link,
                                                  KDL::Tree& tree){
        KDL::Chain chain;
        if (!tree.getChain(root_link, end_link, chain)){
            RCLCPP_ERROR_STREAM(rclcpp::get_logger("get_fk_solver"),
                                "Failed to get KDL chain from tree:");
            RCLCPP_ERROR_STREAM(rclcpp::get_logger("get_fk_solver"),
                                "  " << root_link << " --> " << end_link);

            return KDL::ChainFkSolverPos_recursive(chain);
        }
    }


    /**
     * Checks for self-collision among a vector of collidable parts (robot).
     * This function (without parallelization) determines if any of the
     * collidable parts in the given vector collide with each other, 
     * considering their respective radius. The parts are assumed to be 
     * represented as spheres in 3D space.
     *
     * @param parts: A vector of pointers to ICollidable objects representing 
     *               the robot parts to be checked for self-collision.
     * @return: Returns true if any of the parts collide with each other, 
     *          otherwise false.
     * @throws: This function does not throw any exceptions.
     */
    bool auto_self_collision_check(std::vector<ICollidable*> parts){
        for (auto& part : parts){
            double r_part = (dynamic_cast<RRT::Sphere*>(part))->r;
            for (auto& own_part : parts){
                double r_own_part = (dynamic_cast<RRT::Sphere*>(own_part))->r;
                if (part != own_part){
                    auto distance = (part->frame.p - own_part->frame.p).Norm(); 

                    if (distance <= r_part + r_own_part)
                        return true;
                }
            }
        }
        return false;
    }


(... Example Code Ends ...)

(... Confidencial Code Continues ...)