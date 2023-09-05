/**
 * @author Valerio Di Ceglie
 */

#include "test_base.h"

bool test_base(void) {
    // TODO: implement a csv reading for larger dataset
    // Define dumb dataset
    float points[8][2] = {
        {2.0f, 10.0f},
        {2.0f, 5.0f},
        {8.0f, 4.0f},
        {5.0f, 8.0f},
        {7.0f, 5.0f},
        {6.0f, 4.0f},
        {1.0f, 2.0f},
        {4.0f, 9.0f}
    };

    unsigned int num_points = sizeof (points) / sizeof (points[0]);
    unsigned int dimension = sizeof(points[0]) / sizeof (points[0][0]);
    unsigned int k = 2;

    kmedoid_config config;

    float **objs = (float **) malloc(sizeof (float *) * num_points);
    Cluster *clusters = (Cluster *) malloc(sizeof (Cluster)  * k);

    for(unsigned i = 0; i < num_points; i++) {
        objs[i]  = (float *) malloc(sizeof(float ) * dimension);
    }

    for (unsigned i = 0; i < num_points; i++) {
        for(unsigned int j = 0; j < dimension; j++) {
            objs[i][j] = points[i][j];
        }
    }

    config.k = k;
    config.objs =  objs;
    config.clusters = clusters;
    config.num_objs = num_points;
    config.max_iters = 300;
    config.distance_type = 0;
    config.dim = dimension;
    config.init_method = 2;
    config.method = 1;

    kmedoid(&config);

    // Print the final clusters
    printf("Clustering algorithm ended in %d iterations\n", config.iters);
    printf("Final Clusters:\n");
    print_clusters(&config);

    // Print the silhouette score
    printf("Average Silhouette score: %.5lf\n", config.silhouette_score);

    FILE *fptr;
    fptr= fopen("./cluster_results_2.txt","w");
    if(fptr == NULL) { exit(1); }
    for(unsigned i = 0; i < num_points; i++) {
        unsigned int data_cluster_idx = get_cluster_idx(&config, i);
        fprintf(fptr, "%d,", data_cluster_idx);
    }

    for(unsigned i = 0; i < k; i++) {
        fprintf(fptr, "Cluster %d: medoid index(%d) data points:\n", i, clusters[i].medoid_index);
        for(unsigned j = 0; j < clusters[i].num_points; j++) {
            fprintf(fptr, "%d, ", clusters[i].data_points[j]);
        }
        fprintf(fptr, "\n");
    }
    fclose(fptr);

    // free memory
    kmedoid_deinit(&config);

    for(unsigned i = 0; i < num_points; i++) {
        free(objs[i]);
    }

    free(objs);

    return true;
}

// here we can define new tests


