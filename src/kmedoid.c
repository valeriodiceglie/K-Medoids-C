/**
* @author Valerio Di Ceglie
*/

#include "kmedoid.h"

/* --- Private declaration --- */

/**
 * @brief Compute the euclidean distance between two vectors
 * @param config the configuration instance
 * @param a the first vector
 * @param b the second vector
 * @return the euclidean distance
 */
static float distance(kmedoid_config *config, float *a, float *b);

/**
 * @brief Initialize clusters with given initialization method
 * @param config the configuration instance
 */
static void initialize_clusters(kmedoid_config *config);

/**
 * @brief Assign each point to the nearest cluster
 * @param config the configuration instance
 */
static void assign_points_to_clusters(kmedoid_config *config);

/**
 * @brief Updates the medoid of each cluster using the selected method
 * @param config
 */
static void update_medoids(kmedoid_config *config);

/**
 * @brief Check if the clusters have changed, by comparing their old medoids with the corresponding
 * current medoids
 * @param config the configuration instance
 * @param old_clusters the previous clusters
 * @return true if the clusters have changed, false otherwise
 */
static bool clusters_changed(kmedoid_config *config, unsigned int *old_medoid_idxs);

/**
 * @brief Compute the silhouette score for the resulting clusters
 * @param config the configuration instance
 * @return the average score
 */
static float silhouette_score(kmedoid_config *config);

/**
 * @brief Check if a datapoint is a medoid
 * @param config the configuration instance
 * @param idx_datapoint the index of the datapoint
 * @return true if the datapoint is a ssigned medoid, false otherwise
 * */
static bool is_medoid(kmedoid_config *config, unsigned int i);

/**
 * @brief Compute the total cost for the current configuration
 * @param config the configuration instance
 * @return the cost value
 * */
static float compute_total_cost(kmedoid_config *config);

/* --- private implementation ---*/

static void initialize_clusters(kmedoid_config *config) {

    init_type init_cond = config->init_method;

    // Seed the random number generator (in case of UNIX like system, use the /dev/rand file)
    #ifdef _WIN32
        srand(time(NULL));
    #endif

    #ifdef __unix__
        unsigned int seed;
        FILE *f;
        f = fopen("/dev/random", "r");
        fread(&seed, sizeof(seed), 1, f);
        fclose(f);
        srand(seed);
    #endif

    //                                  //
    //        RANDOM DATAPOINT          //
    //                                  //
    if(init_cond == RANDOM_DATAPOINT) {
        for(unsigned int i = 0; i < config->k; i++) {
            config->clusters[i].medoid = malloc(sizeof(float) * config->dim);
            unsigned int random_index     = (unsigned int) rand() / (RAND_MAX / (config->num_objs));
            memcpy(config->clusters[i].medoid, config->objs[random_index], sizeof(float) * config->dim);
            config->clusters[i].medoid_index = random_index;
            config->clusters[i].num_points   = 0;
            config->clusters[i].data_points  = (unsigned int *) malloc(sizeof(unsigned int) *
                                                                   config->num_objs);
        }
    }

    //                      //
    //      K-MEDOIDS++     //
    //                      //
    else if(init_cond == KMEDOIDSPP) {
        unsigned int n_clusters  = config->k;
        unsigned int sample_size = config->num_objs;
        unsigned int dimension = config->dim;

        // Randomly select the first medoid
        unsigned int first_medoid_idx       = rand() % sample_size;
        config->clusters[0].medoid_index = first_medoid_idx;
        config->clusters[0].medoid = malloc(sizeof(float) * dimension);
        memcpy(config->clusters[0].medoid, config->objs[first_medoid_idx],
               sizeof(float) * dimension);
        config->clusters[0].num_points = 0;
        config->clusters[0].data_points = malloc(sizeof(unsigned int) * sample_size);

        unsigned int n_local_trials = 2 + (int) log(n_clusters);

        /*
        * distance_to_closest     -> Array to store the squared distances to the closest medoid for
        *                            each point
        * distances_to_candidates -> Matrix for the squared distances to the medoid candidates
        * new_dist                -> Array to store the minimum between the closest medoid distance
        *                            and the corresponding distance to medoid candidates
        * rand_val                -> Array to store the generated random value
        * medoid_idx              -> Array to store the medoid indexes, which are selected with a
        *                            probability distribution proportional to the squared distances
        *                            of the previous selected medoids
        */

        float *distances_to_closest = (float *) malloc(sizeof(float) * sample_size);
        float **distances_to_candidates = (float **) malloc(sizeof(float *) *
                                                                      sample_size);
        float *new_dist = (float *) malloc(sizeof(float) * sample_size);
        float new_pot;
        float *rand_val  = malloc(sizeof(float) * n_local_trials);
        unsigned int *medoid_idx = malloc(sizeof(unsigned int) * n_local_trials);

        float sum_distances = 0.0f;

        /*
         * Initialize distances to closest medoid (that is the distance between the datapoint and
         * the first medoid)
        */
        for(unsigned int i = 0; i < sample_size; i++) {
            distances_to_candidates[i] = malloc(sizeof(float) * n_local_trials);
            distances_to_closest[i] = distance(config, config->clusters[0].medoid, config->objs[i]) *
                                      distance(config, config->clusters[0].medoid, config->objs[i]);
            sum_distances += distances_to_closest[i];
        }

        // Select the remaining medoids
        for(unsigned int i = 1; i < n_clusters; i++) {
            for(unsigned int k = 0; k < n_local_trials; k++) {
                rand_val[k] = ((float) rand() / (float) (RAND_MAX)) * sum_distances;
            }

            // Compute cumulative sum and search for candidate indexes

            //printf("current sum distances: %.3lf\n", sum_distances);
            for(unsigned int k = 0; k < n_local_trials; k++) {
                float cumulative_distances = 0.0f;
                //printf("rand val[%d]: %.3lf ", k, rand_val[k]);
                for(unsigned int j = 0; j < sample_size; j++) {
                    cumulative_distances += distances_to_closest[j];
                    if(cumulative_distances > rand_val[k]) {
                        medoid_idx[k] = j;
                        //printf(" --- index datapoint as potential medoid: %d\n", j);
                        break;
                    }
                }
            }

            // Compute squared distances to medoid candidates
            for(unsigned int j = 0; j < sample_size; j++) {
                for(unsigned int k = 0; k < n_local_trials; k++) {
                    distances_to_candidates[j][k] = distance(config, config->objs[medoid_idx[k]],
                                                             config->objs[j]) *
                                                    distance(config, config->objs[medoid_idx[k]],
                                                             config->objs[j]);
                }
            }

            unsigned int best_candidate;
            float best_pot          = FLT_MAX;
            float *best_distance_sq = malloc(sizeof(float) * sample_size);

            for(unsigned int trial = 0; trial < n_local_trials; trial++) {
                new_pot = 0.0f;

                for(unsigned int j = 0; j < sample_size; j++) {
                    new_dist[j] = fminf(distances_to_closest[j], distances_to_candidates[j][trial]);
                    new_pot += new_dist[j];
                }

                if(new_pot < best_pot) {
                    best_pot       = new_pot;
                    best_candidate = medoid_idx[trial];
                    memcpy(best_distance_sq, new_dist, sizeof(float) * sample_size);
                }
            }

            //printf("Selected index medoid: %d\n", best_candidate);
            sum_distances = best_pot;
            memcpy(distances_to_closest, best_distance_sq, sizeof(float) * sample_size);

            config->clusters[i].medoid_index = best_candidate;
            config->clusters[i].medoid = malloc(sizeof(float) * dimension);
            memcpy(config->clusters[i].medoid, config->objs[best_candidate], sizeof(float) * dimension);
            config->clusters[i].num_points = 0;
            config->clusters[i].data_points = malloc(sizeof(unsigned int) * sample_size);
            free(best_distance_sq);
        }

        // Cleanup allocated memory
        for(unsigned int i = 0; i < sample_size; i++) {
            free(distances_to_candidates[i]);
        }

        free(rand_val);
        free(medoid_idx);
        free(distances_to_closest);
        free(distances_to_candidates);
        free(new_dist);

    }
    //                      //
    //         BUILD        //
    //                      //
    else if(init_cond == BUILD) {
        /*
        * Compute the sum of distances of each data point to x_i data point and selecting as medoid
        * the i-one which retain the smallest distance sum
        */
        float *distances             = (float *) malloc(sizeof(float) * config->num_objs);
        float *distance_nearest = malloc(sizeof(float) * config->num_objs);

        for(unsigned int i = 0; i < config->num_objs; i++) {
            for(unsigned int j = 0; j < config->num_objs; j++) {
                if(i == j)
                    continue;
                distances[i] += distance(config, config->objs[i], config->objs[j]);
            }
        }

        float min_distance = FLT_MAX;
        unsigned int first_medoid_idx;

        for(unsigned int i = 0; i < config->num_objs; i++) {
            if(distances[i] < min_distance) {
                min_distance     = distances[i];
                first_medoid_idx = i;
            }
        }

        config->clusters[0].medoid_index = first_medoid_idx;
        config->clusters[0].medoid = malloc(sizeof(float) * config->dim);
        memcpy(config->clusters[0].medoid, config->objs[first_medoid_idx], sizeof(float)*config->dim);
        config->clusters[0].num_points = 0;
        config->clusters[0].data_points = malloc(sizeof(unsigned int) * config->num_objs);
        free(distances);

        // Set distance to nearest medoid
        for(unsigned int i = 0; i < config->num_objs; i++) {
            distance_nearest[i] = distance(config, config->objs[i], config->clusters[0].medoid);
        }

        unsigned int number_curr_medoids = 1;
        float cost_change;
        float cost_change_max;
        unsigned int new_medoid_idx;

        for(unsigned int i = 1; i < config->k; i++) {
            cost_change_max = 0.0f;
            for(unsigned int j = 0; j < config->num_objs - number_curr_medoids; j++) {
                if(is_medoid(config, j)) {
                    continue;
                }
                cost_change = 0.0f;
                for(unsigned int k = 0; k < config->num_objs - number_curr_medoids; k++) {
                    if(is_medoid(config, k)) {
                        continue;
                    }
                    float d_k = distance_nearest[k];
                    cost_change += fmaxf(0, d_k - distance(config, config->objs[j],
                                                          config->objs[k]));
                }
                if(cost_change >= cost_change_max) {
                    cost_change_max = cost_change;
                    new_medoid_idx = j;
                }
            }

            number_curr_medoids++;

            for(unsigned int j = 0; j < config->num_objs; j++) {
                float curr_distance_nearest = distance_nearest[j];
                distance_nearest[j] = fminf(curr_distance_nearest,
                                            distance(config, config->objs[j],
                                                     config->objs[new_medoid_idx]));
            }

            config->clusters[i].medoid_index = new_medoid_idx;
            config->clusters[i].medoid = malloc(sizeof(float) * config->dim);
            memcpy(config->clusters[i].medoid, config->objs[new_medoid_idx], sizeof(float) * config->dim);
            config->clusters[i].num_points = 0;
            config->clusters[i].data_points = malloc(sizeof(unsigned int) * config->num_objs);
        }
        free(distance_nearest);
    }
}

static bool is_medoid(kmedoid_config *config, unsigned int idx){
    for(unsigned int i = 0; i < config->k; i++) {
        if(config->clusters[i].medoid_index == idx)
            return true;
    }
    return false;
}

static float distance(kmedoid_config *config, float *a, float *b){

    unsigned int distance_type = config->distance_type;
    unsigned int dimension = config->dim;
    float distance = 0;

    if(distance_type == EUCLIDEAN) {
        for(unsigned int i = 0; i < dimension; i++) {
            float x1 = a[i];
            float x2 = b[i];
            distance += (x1 - x2) * (x1 - x2);
        }
        return sqrtf(distance);
    }

    else if(distance_type == MANHATTAN) {
        for(unsigned int i = 0; i < dimension; i++) {
            float x1 = a[i];
            float x2 = b[i];
            distance += fabsf(x1 - x2);
        }
        return distance;
    }
}

static void assign_points_to_clusters(kmedoid_config *config) {
    bool *point_assigned = (bool *) malloc(sizeof(bool) * config->num_objs);

    for(unsigned int i = 0; i < config->num_objs; i++) {
        point_assigned[i] = false;
    }
    for(unsigned int i = 0; i < config->k; i++) {
        config->clusters[i].num_points = 0;

        // Free old data points
        free(config->clusters[i].data_points);

        // Allocate memory for new data points
        config->clusters[i].data_points = (unsigned int *) malloc(sizeof(unsigned int) * config->num_objs);
    }

    for(unsigned int i = 0; i < config->num_objs ; i++) {
        if(point_assigned[i])
            continue;

        float min_distance = FLT_MAX;
        unsigned int cluster_index = 0;

        for(unsigned int j = 0; j < config->k; j++) {
            float d = distance(config, config->objs[i], config->clusters[j].medoid);
            if(d < min_distance) {
                min_distance  = d;
                cluster_index = j;
            }
        }
        config->clusters[cluster_index].data_points[config->clusters[cluster_index].num_points] = i;
        config->clusters[cluster_index].num_points += 1;

        point_assigned[i] = true;
    }
    free(point_assigned);
}

static void update_medoids(kmedoid_config *config)
{
    method_type method   = config->method;

    /*
     * NOTE: the K-Medoids algorithm with random initialization and alternate method could lead to
     *       empty clusters
     */

    //                      //
    //      ALTERNATE       //
    //                      //
    if(method == ALTERNATE) {

        float *min_cost = malloc(sizeof (float) * config->k);
        // Compute the current cost for each medoid
        for (unsigned int i = 0; i < config->k; i++) {
            for (unsigned int j = 0; j < config->clusters[i].num_points; j++) {
                min_cost[i] += distance(config, config->clusters[i].medoid,
                                     config->objs[config->clusters[i].data_points[j]]);
            }
        }

        for (unsigned int i = 0; i < config->k; i++) {
            unsigned int medoid_index = config->clusters[i].medoid_index;
            for (unsigned int j = 0; j < config->clusters[i].num_points; j++) {
                float cost = 0.0f;
                for (unsigned int k = 0; k < config->clusters[i].num_points; k++) {
                    if (j == k)
                        continue;
                    cost += distance(config, config->objs[config->clusters[i].data_points[j]],
                                     config->objs[config->clusters[i].data_points[k]]);
                }
                if (cost < min_cost[i]) {
                    min_cost[i]     = cost;
                    medoid_index = config->clusters[i].data_points[j];
                }
            }

            config->clusters[i].medoid_index = medoid_index;
            memcpy(config->clusters[i].medoid, config->objs[medoid_index], sizeof(float) * config->dim);
        }

        free(min_cost);
    }

    /* TODO: to improve the performance of the PAM algorithm, we could store the distances matrix
     *       for all the datapoints in the clustering space
     * */

    //              //
    //      PAM     //
    //              //
    else if(method == PAM) {

        float **distances = (float **) malloc(sizeof(float *) * config->k);

        /*                          DISTANCES MATRIX
         *
         *          |datapoint_0   |  datapoint_1  | ... | datapoint_n  |
         * _________|______________|_______________|_____|_____________ |
           medoid_0 | d(m_0,dp_0)  |  d(m_0,dp_1)  | ... | d(m_0,dp_n   |
           ---------|--------------|---------------|-----|------------- |
           medoid_1 | d(m_1,dp_0)  |  d(m_1,dp_1)  | ... |              |
           ---------|--------------|---------------|-----|------------- |
              .            .               .                    .
              .            .               .         ...        .
              .            .               .                    .
           ---------|--------------|---------------|-----|------------- |
           medoid_k | d(m_k,dp_0)  | d(m_k,dp_1    | ... | d(m_k,dp_n)  |
           _________|______________|_______________|_____|______________|
         */

        // Compute the distance matrix to the first and second-closest points among medoids
        for(unsigned int i = 0; i < config->k; i++) {
            distances[i] = malloc(sizeof(float) * config->num_objs);
            for(unsigned int j = 0; j < config->num_objs; j++) {
                distances[i][j] = distance(config, config->clusters[i].medoid, config->objs[j]);
            }
        }

        /*
         * Sort the distance matrix along columns:
         * The first row will contain the distance to nearest;
         * The second row will contain the second distance to nearest.
        */
        for(unsigned int i = 0; i < config->num_objs; i++) {
            for(unsigned int j = 0; j < config->k; j++) {
                for(unsigned int k = j + 1; k < config->k; k++) {
                    if (distances[j][i] > distances[k][i]) {
                        float temp = distances[j][i];
                        distances[j][i] = distances[k][i];
                        distances[k][i] = temp;
                    }
                }
            }
        }

        float cost_change;
        float best_cost_change = 0.0f;
        unsigned int medoid_idx, datapoint_idx;
        bool cluster_i_bool, not_cluster_i_bool, second_best_medoid, not_second_best_medoid;

        // Compute the change in cost for each swap
        for(unsigned int h = 0; h < config->num_objs; h++) {
            if (is_medoid(config, h)) { continue; }
            for(unsigned int i = 0; i < config->k; i++) {
                cost_change = 0.0f;
                for(unsigned int j = 0; j < config->num_objs; j++) {
                    if (is_medoid(config, j))  { continue; }
                    // d_nearest(j) == d(m_i, x_j)
                    cluster_i_bool = (distances[0][j] ==
                                     distance(config, config->clusters[i].medoid,
                                               config->objs[j]));
                    // d_nearest(j) != d(m_i, x_j)
                    not_cluster_i_bool = (distances[0][j] !=
                                          distance(config, config->clusters[i].medoid,
                                                   config->objs[j]));
                    // d_second(j) > d(x_h, x_j)
                    second_best_medoid = (distances[1][j] >
                                          distance(config, config->objs[h], config->objs[j]));
                    // d_second(j) <= d(x_h, x_j)
                    not_second_best_medoid = (distances[1][j] <=
                                              distance(config, config->objs[h], config->objs[j]));

                    if (cluster_i_bool && second_best_medoid) {
                        cost_change += distance(config, config->objs[j], config->objs[h]) -
                                       distances[0][j];
                    }
                    else if(cluster_i_bool && not_second_best_medoid) {
                        cost_change += distances[1][j] - distances[0][j];
                    }
                    else if(not_cluster_i_bool &&
                            (distance(config, config->objs[j], config->objs[h])
                             < distances[0][j])) {
                        cost_change += distance(config, config->objs[j], config->objs[h]) -
                                       distances[0][j];
                    }
                }

                second_best_medoid = distance(config, config->objs[h], config->clusters[i].medoid)
                                   < distances[1][i];
                if (second_best_medoid) {
                    cost_change += distance(config, config->clusters[i].medoid, config->objs[h]);
                }
                else { cost_change +=  distances[1][config->clusters[i].medoid_index]; }

                if (cost_change < best_cost_change) {
                    best_cost_change = cost_change;
                    medoid_idx = i;
                    datapoint_idx = h;
                }
            }
        }
        if(best_cost_change < 0) {
            config->clusters[medoid_idx].medoid_index = datapoint_idx;
            memcpy(config->clusters[medoid_idx].medoid, config->objs[datapoint_idx],
                   sizeof(float) * config->dim);
        }
        for(unsigned int i = 0; i < config->k; i++) {
            free(distances[i]);
        }
        free(distances);
    }
}

static float compute_total_cost(kmedoid_config *config) {
    float total_cost = 0.0f;
    unsigned int n_clusters  = config->k;

    for(unsigned int i = 0; i < n_clusters; i++) {
        unsigned int medoid_idx = config->clusters[i].medoid_index;

        for(unsigned int j = 0; j < config->clusters[i].num_points; j++) {
            unsigned int point_idx = config->clusters[i].data_points[j];
            float dist = distance(config, config->objs[point_idx], config->objs[medoid_idx]);
            total_cost += dist;
        }
    }

    return total_cost;
}

static bool clusters_changed(kmedoid_config *config, unsigned int *old_medoid_idxs)
{
    for(unsigned int i = 0; i < config->k; i++) {
        if(config->clusters[i].medoid_index != old_medoid_idxs[i]) {
            return true;
        }
    }
    return false;
}

static float silhouette_score(kmedoid_config *config) {

    // Compute the average silhouette score over all samples
    Cluster *curr_cluster;
    unsigned int curr_cluster_index;
    float sum_a, sum_b, a, temp_b, s;
    float b = FLT_MAX;

    // Compute s foreach data point
    for (unsigned int i = 0; i < config->num_objs ; i++) {

        // Find the current data point's cluster
        curr_cluster_index = get_cluster_idx(config, i);
        curr_cluster = &config->clusters[curr_cluster_index];


        // Compute intra-cluster distance sum
        sum_a = 0;
        for(unsigned int j = 0; j < curr_cluster->num_points; j++) {

            a = 0;
            if(curr_cluster->data_points[j] == i)
                continue ;
            sum_a += distance(config, config->objs[i], config->objs[curr_cluster->data_points[j]]);
        }
        a = sum_a / ((curr_cluster->num_points) -1);


        // Compute inter-cluster distance sum
        sum_b = 0.0f;
        b = FLT_MAX;
        for(unsigned int j = 0; j < config->k; j++) {

            if (j == curr_cluster_index)
                continue ;
            for(unsigned int k = 0; k < config->clusters[j].num_points; k++) {
                sum_b += distance(config, config->objs[i],
                                  config->objs[config->clusters[j].data_points[k]]);
            }
            temp_b = sum_b / config->clusters[j].num_points;
            if(temp_b < b) {
                b = temp_b;
            }
        }
        s += (b - a) / fmaxf(b, a);
    }

    return s / config->num_objs;
}

/* --- public implementation --- */

void print_clusters(kmedoid_config *config){
    for (unsigned int i = 0; i < config->k; i++) {
        printf("Cluster %d: Medoid: [", i);
        for(unsigned int j = 0; j < config->dim; j++) {
            printf(" %.3lf ", config->clusters[i].medoid[j]);
        }
        printf("]  -  medoid index (%d)\n", config->clusters[i].medoid_index);
        printf("number of points: %d\n", config->clusters[i].num_points);
        /*printf("Data points: \n");
        for(unsigned int j = 0; j < config->clusters[i].num_points; j++) {
            unsigned int point_index = config->clusters[i].data_points[j];
            printf("index %d: ( ", point_index);
            uvec_foreach(float, config->objs[point_index], datapoint){
                printf("%.2lf ", *datapoint.item);
            }
            printf(")\n");
        }*/
    }
}

void kmedoid(kmedoid_config *config){

    bool clusters_updated = true;
    unsigned int iterations = 0;

    initialize_clusters(config);

    printf("Initial Cluster: \n");
    print_clusters(config);

    unsigned int *old_medoid_idxs = (unsigned int *) malloc(sizeof(unsigned int) * config->k);

    while (clusters_updated && iterations < config->max_iters) {
        // Assign points to clusters
        assign_points_to_clusters(config);

        // Store old clusters
        for (unsigned int i = 0; i < config->k; i++) {
            old_medoid_idxs[i] = config->clusters[i].medoid_index;
        }

        // Update medoids
        update_medoids(config);

        for (unsigned int k = 0; k < config->k; k++) {
            if(config->clusters[k].num_points == 0) {
                config->clusters[k].medoid_index = old_medoid_idxs[k];
                memcpy(config->clusters[k].medoid, config->objs[old_medoid_idxs[k]],
                       sizeof(float) * config->dim);
                assign_points_to_clusters(config);
            }
        }

        // Check if clusters have changed
        clusters_updated = clusters_changed(config, old_medoid_idxs);

        // Increment iterations
        iterations++;
    }

    config->iters = iterations;
    config->silhouette_score = silhouette_score(config);

    free(old_medoid_idxs);
}

unsigned int get_cluster_idx(kmedoid_config *config, unsigned int idx_datapopint) {
    for(unsigned int i = 0; i < config->k ; i++) {
        for(unsigned int j = 0; j < config->clusters[i].num_points; j++) {
            if(config->clusters[i].data_points[j] == idx_datapopint)
                return i;
        }
    }
}

void kmedoid_deinit(kmedoid_config *config) {
    // Free array of indexes data points
    for (unsigned int i = 0; i < config->k; i++) {
        free(config->clusters[i].data_points);
        free(config->clusters[i].medoid);
    }
}
