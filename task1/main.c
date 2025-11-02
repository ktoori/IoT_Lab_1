#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int comm_sz;
int my_rank;

long long RandomPoints(long long local_n) {
    long long count = 0;
    double x, y, distance;

    srand(time(NULL) + my_rank * 1000);

    for (long long i = 0; i < local_n; i++) {
        x = -1.0 + 2.0 * ((double)rand() / RAND_MAX);
        y = -1.0 + 2.0 * ((double)rand() / RAND_MAX);

        distance = x * x + y * y;

        if (distance <= 1.0) {
            count++;
        }
    }

    return count;
}

double ComputePi(long long total_n) {
    long long local_n = total_n / comm_sz;

    if (my_rank == 0) {
        local_n += total_n % comm_sz;
    }

    long long local_count = RandomPoints(local_n);
    long long global_count = 0;

    MPI_Reduce(&local_count, &global_count, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    double pi_estimate = 0.0;
    if (my_rank == 0) {
        pi_estimate = 4.0 * global_count / total_n;
    }

    return pi_estimate;
}

double ComputePiSimple(long long total_n) {
    
    long long count = 0;
    double x, y, distance;

    srand(time(NULL) + 1000);

    for (long long i = 0; i < total_n; i++) {
        x = -1.0 + 2.0 * ((double)rand() / RAND_MAX);
        y = -1.0 + 2.0 * ((double)rand() / RAND_MAX);

        distance = x * x + y * y;

        if (distance <= 1.0) {
            count++;
        }
    }

    double pi_estimate = 4.0 * count / total_n;
    return pi_estimate;

}

int main() {
    int long long n;

    MPI_Init(NULL, NULL);

    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz); 
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (my_rank == 0) {
        printf("Enter number of points: ");
        fflush(stdout);
        scanf("%lld", &n);
        printf("\n");
    }

    MPI_Bcast(&n, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    double start_time, end_time, local_time, global_time, local_time_simple;
    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime();

    double pi_result = ComputePi(n);

    end_time = MPI_Wtime();
    local_time = end_time - start_time;

    MPI_Reduce(&local_time, &global_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        start_time = MPI_Wtime();
        double pi_result = ComputePiSimple(n);
        end_time = MPI_Wtime();
        local_time_simple = end_time - start_time;
    }

    if (my_rank == 0) {
        long long base_per_process = n / comm_sz;
        long long remainder = n % comm_sz;
        float speed_up = local_time_simple / global_time;
        float efficiency = speed_up / comm_sz;

        printf("RESULTS\n");
        printf("Estimated PI value: %.10f\n", pi_result);
        printf("True PI value:      %.10f\n", M_PI);
        printf("\nParallel program time:     %.6f seconds\n", global_time);
        printf("Linear program time:     %.6f seconds\n", local_time_simple);
        printf("\nSpeed-up:     %f \n", speed_up);
        printf("Efficiency:     %f \n", efficiency);

    }

    MPI_Finalize();

    return 0;
}