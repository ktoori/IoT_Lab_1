#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

void fillMatrix(int* matrix, int r, int c) {
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            matrix[i * c + j] = rand() % 10;
        }
    }
}

void fillVector(int* vector, int n) {
    for (int i = 0; i < n; i++) {
        vector[i] = rand() % 10;
    }
}

void multRows(int* matrix, int* vector, int* result, int r, int c) {
    for (int i = 0; i < r; i++) {
        result[i] = 0;
        for (int j = 0; j < c; j++) {
                result[i] += matrix[i * c + j] * vector[j];
        }
    }
}

int main() {
    MPI_Init(NULL, NULL);
    int comm_sz;
    int my_rank;

    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int rows;
    int cols;
    int* matrix = NULL;
    int* vector = NULL;
    int* result = NULL;

    int* sendcounts = NULL;
    int* displs = NULL;
    int* recvcounts = NULL;
    int* recvdispls = NULL;

    double timeStart;
    double timeEnd;
    double localTime;
    double totalTime;


    if (my_rank == 0) {
        scanf("%d %d", &rows, &cols);

        matrix = (int*)malloc(rows * cols * sizeof(int));
        vector = (int*)malloc(cols * sizeof(int));
        result = (int*)malloc(rows * sizeof(int));

        srand(time(0));
        fillMatrix(matrix, rows, cols);
        fillVector(vector, cols);

        sendcounts = (int*)malloc(comm_sz * sizeof(int));
        displs = (int*)malloc(comm_sz * sizeof(int));
        recvcounts = (int*)malloc(comm_sz * sizeof(int));
        recvdispls = (int*)malloc(comm_sz * sizeof(int));

        int baseRows = rows / comm_sz;
        int extraRows = rows % comm_sz;
        int offset = 0;
        for (int i = 0; i < comm_sz; i++) {
            int rowsForProc = baseRows + (i < extraRows ? 1 : 0);

            sendcounts[i] = rowsForProc * cols;
            displs[i] = offset * cols;

            recvcounts[i] = rowsForProc;
            recvdispls[i] = offset;

            offset += rowsForProc;
        }
    }

    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int localRows;
    if (my_rank == 0) {
        for (int i = 0; i < comm_sz; i++) {
            if (i == my_rank) localRows = recvcounts[i];
        }
    }

    MPI_Scatter(recvcounts, 1, MPI_INT, &localRows, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int* localMatrix = (int*)malloc(localRows * cols * sizeof(int));
    int* localResult = (int*)malloc(localRows * sizeof(int));

    if (my_rank != 0) {
        vector = (int*)malloc(cols * sizeof(int));
    }

    MPI_Scatterv(matrix, sendcounts, displs, MPI_INT, localMatrix, localRows * cols, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Bcast(vector, cols, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    timeStart = MPI_Wtime();

    multRows(localMatrix, vector, localResult, localRows, cols);

    timeEnd = MPI_Wtime();
    localTime = timeEnd - timeStart;

    MPI_Gatherv(localResult, localRows, MPI_INT, result, recvcounts, recvdispls, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Reduce(&localTime, &totalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        printf("\nMatrix:\n");
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                printf("%d ", matrix[i * cols + j]);
            }
            printf("\n");
        }

        printf("\nVector:\n");
        for (int i = 0; i < cols; i++) {
            printf("%d\n", vector[i]);
        }

        printf("\nResult:\n");
        for (int i = 0; i < rows; i++) {
            printf("%d\n", result[i]);
        }

        printf("\nExecution time: %.3f ms\n", totalTime * 1000.0);

        free(matrix);
        free(result);
        free(sendcounts);
        free(displs);
        free(recvcounts);
        free(recvdispls);
    }

    free(localMatrix);
    free(vector);
    free(localResult);

    MPI_Finalize();
    return 0;
}