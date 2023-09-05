/** *
* @author Valerio Di Ceglie
*/

#ifndef KMEDOID_H
#define KMEDOID_H
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <limits.h>
#include <math.h>
#include <string.h>
#include <float.h>
#include <time.h>

/**
 * @brief Enumeration for the initialization type
 * @enum RANDOM_DATAPOINT select k random data points as initial medoid
 * @enum KMEDOIDSPP Choose the first medoid uniformly at random. Then, compute the distance squared
 * probabilities for each data point related to the first medoid. Loop for each remaining medoid
 * @enum BUILD initialization method for PAM
 * */
typedef enum initType_e {
    RANDOM_DATAPOINT = 0,
    KMEDOIDSPP = 1,
    BUILD = 2
}init_type;

/**
 * @brief Enumeration for the type of distance to compute
 * @enum EUCLIDEAN
 * @enum MANHATTAN
 * */
typedef enum DistanceType_e {
    EUCLIDEAN = 0,
    MANHATTAN = 1
} distance_type;

/**
 * @brief Enumeration for the algorithm type
 *
 *  @enum ALTERNATE faster but less accurate
 *  @enum PAM higher computational cost but more accurate
 *  @note the PAM type must have a greedy initialization (BUILD)
 * */
typedef enum MethodType_e {
    ALTERNATE = 0,
    PAM = 1
} method_type;

/**
 * @brief Structure to represent a cluster
 * */
typedef struct Cluster_s {
    /** An array containing the medoid coordinates */
    float *medoid;

    /** An array that contains data point indexes belonging to corresponding cluster */
    unsigned int *data_points;

    unsigned int medoid_index;

    /** Number of points that belong to cluster*/
    unsigned int num_points;
} Cluster;

/**
 * @brief Struct for the K-Medoids initial configuration
 * */
typedef struct KmedoidConfig_s {
    /** The number of cluster */
    unsigned int k;

    /** The number of objects in the clustering space*/
    unsigned int num_objs;

    /** The number of columns in the clustering space*/
    unsigned int dim;

    /** Array of pointers to the clustering space */
    float **objs;

    /** Array of clusters */
    Cluster *clusters;

    /** Numbers of iteration to reach a stable solution */
    unsigned int iters;

    /** The number of maximum iterations */
    unsigned int max_iters;

    /** Silhouette score for the current configuration*/
    float silhouette_score;

    /** Initialization method
     * @enum 0 --- RANDOM DATAPOINT
     * @enum 1 --- K-MEDOIDS++
     * @enum 2 --- BUILD
     * */
    unsigned int init_method;

    /**
     * @brief K-Medoids algorithm type
     * @enum 0 --- ALTERNATE
     * @enum 1 --- PAM
     * @note ALTERNATE algorithm with RANDOM DATAPOINT initialization could lead to empty clusters,
     * while using a Training Matrix as dataset
     * */
    unsigned int method;

    /** @brief Distance type
     * @enum 0 --- EUCLIDEAN
     * @enum 1 --- MANHATTAN
     */
    unsigned int distance_type;
} kmedoid_config;

/* --- public declaration --- */

/**
 * @brief Print the cluster medoids and his data points
 *
 * @param config the configuration instance
 */
void print_clusters(kmedoid_config *config);

/**
 * @brief Compute the K-Medoids algorithm
 * @param config the configuration instance
 */
void kmedoid(kmedoid_config *config);

/**
 * @brief Get the cluster id which detain the datapoint
 * @param config the configuration instance
 * @param idx_datapoint the datapoint index
 * @return The cluster id which detain the datapoint
 * */
unsigned int get_cluster_idx(kmedoid_config *config, unsigned int idx_datapoint);

/**
 * @brief Deinitialize the K-Medoids configuration
 * @param config the configuration instance
 * @note User must deinit the clusters instances array and the objects array
 */
void kmedoid_deinit(kmedoid_config *config);

#endif // KMEDOID_H
