#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <time.h>
#include <string.h>
#define MATRIX_SIZE 600
#define EPSILON 1e-8
#define DEBUG 0

void initialize_matrix(double *matrix, int size) {
    for (int i = 0; i < size * size; i++) {
        matrix[i] = (double)(rand() % 10);
    }
}

void zero_matrix(double *matrix, int size) {
    memset(matrix, 0, size * size * sizeof(double));
}

void print_matrix(double *matrix, int size, const char *name) {
    if (!DEBUG || size > 8) return;  

    printf("\n%s (размер %d×%d):\n", name, size, size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            printf("%7.1f ", matrix[i * size + j]);
        }
        printf("\n");
    }
}

void block_multiply(double *A_block, double *B_block, double *C_block, 
                    int block_size) {
    for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
            for (int k = 0; k < block_size; k++) {
                C_block[i * block_size + j] += 
                    A_block[i * block_size + k] * B_block[k * block_size + j];
            }
        }
    }
}

void sequential_multiply(double *A, double *B, double *C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            C[i * size + j] = 0.0;
            for (int k = 0; k < size; k++) {
                C[i * size + j] += A[i * size + k] * B[k * size + j];
            }
        }
    }
}

double verify_result(double *C_parallel, double *C_sequential, int size) {
    double max_error = 0.0;
    for (int i = 0; i < size * size; i++) {
        double error = fabs(C_parallel[i] - C_sequential[i]);
        if (error > max_error) max_error = error;
    }
    return max_error;
}

int main(int argc, char *argv[]) {
    int rank, size, grid_size, coords[2];
    int dims[2], periods[2] = {1, 1}, reorder = 1;
    MPI_Comm cart_comm;
    double *A_full = NULL, *B_full = NULL, *C_full = NULL;
    double *A_block, *B_block, *C_block;
    int block_size;
    double start_time, end_time, parallel_time, seq_time;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    grid_size = (int)sqrt((double)size);
    if (grid_size * grid_size != size) {
        if (rank == 0) {
            fprintf(stderr, "ОШИБКА: Количество процессов должно быть точным квадратом!\n");
            fprintf(stderr, "Допустимые значения: 1, 4, 9, 16, 25, 36, 49, 64...\n");
            fprintf(stderr, "Вы задали: %d процессов\n", size);
        }
        MPI_Finalize();
        return 1;
    }

    if (MATRIX_SIZE % grid_size != 0) {
        if (rank == 0) {
            fprintf(stderr, "ОШИБКА: Размер матрицы (%d) должен делиться на √(процессов) = %d\n", MATRIX_SIZE, grid_size);
        }
        MPI_Finalize();
        return 1;
    }

    block_size = MATRIX_SIZE / grid_size;

    if (rank == 0) {
        printf("\n");
        
    }

    dims[0] = dims[1] = grid_size;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    if (DEBUG && rank == 0) {
        printf("[DEBUG] Создана декартова топология %d×%d\n", dims[0], dims[1]);
    }

    A_block = (double*)malloc(block_size * block_size * sizeof(double));
    B_block = (double*)malloc(block_size * block_size * sizeof(double));
    C_block = (double*)calloc(block_size * block_size, sizeof(double));

    if (!A_block || !B_block || !C_block) {
        fprintf(stderr, "Процесс %d: ошибка выделения памяти для блоков\n", rank);
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        A_full = (double*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
        B_full = (double*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(double));
        C_full = (double*)calloc(MATRIX_SIZE * MATRIX_SIZE, sizeof(double));
        if (!A_full || !B_full || !C_full) {
            fprintf(stderr, "Ошибка выделения памяти для полных матриц\n");
            MPI_Finalize();
            return 1;
        }
        srand(time(NULL) + rank);

        if (rank == 0) 
            printf("[%d] Генерирование матриц A и B...\n", rank);

        initialize_matrix(A_full, MATRIX_SIZE);
        initialize_matrix(B_full, MATRIX_SIZE);

        print_matrix(A_full, MATRIX_SIZE, "Матрица A");
        print_matrix(B_full, MATRIX_SIZE, "Матрица B");

        if (rank == 0) 
            printf("[%d] Матрицы готовы\n", rank);
    }
  
    MPI_Datatype block_type, resized_block;
    MPI_Type_vector(block_size, block_size, MATRIX_SIZE, 
                    MPI_DOUBLE, &block_type);
    MPI_Type_create_resized(block_type, 0, sizeof(double), &resized_block);
    MPI_Type_commit(&resized_block);

    int *sendcounts = NULL, *displs = NULL;
    if (rank == 0) {
        sendcounts = (int*)malloc(size * sizeof(int));
        displs = (int*)malloc(size * sizeof(int));


        for (int i = 0; i < grid_size; i++) {
            for (int j = 0; j < grid_size; j++) {
                int proc_rank = i * grid_size + j;
                sendcounts[proc_rank] = 1;
         
                displs[proc_rank] = i * MATRIX_SIZE * block_size + j * block_size;
            }
        }
    }

    if (rank == 0) printf("[%d] Распределение блоков матрицы A...\n", rank);
    MPI_Scatterv(A_full, sendcounts, displs, resized_block,
                 A_block, block_size * block_size, MPI_DOUBLE,
                 0, cart_comm);

    if (rank == 0) printf("[%d] Распределение блоков матрицы B...\n", rank);
    MPI_Scatterv(B_full, sendcounts, displs, resized_block,
                 B_block, block_size * block_size, MPI_DOUBLE,
                 0, cart_comm);

    if (DEBUG && rank == 0) {
        printf("[DEBUG] Блоки распределены\n");
    }


    int src_rank, dst_rank;

    if (rank == 0) printf("[%d] Выполнение предварительного сдвига A влево на %d позиций...\n", 
                          rank, coords[0]);
    MPI_Cart_shift(cart_comm, 1, -coords[0], &src_rank, &dst_rank);
    MPI_Sendrecv_replace(A_block, block_size * block_size, MPI_DOUBLE,
                         dst_rank, 0, src_rank, 0, cart_comm, &status);

    if (rank == 0) printf("[%d] Выполнение предварительного сдвига B вверх на %d позиций...\n", 
                          rank, coords[1]);
    MPI_Cart_shift(cart_comm, 0, -coords[1], &src_rank, &dst_rank);
    MPI_Sendrecv_replace(B_block, block_size * block_size, MPI_DOUBLE,
                         dst_rank, 0, src_rank, 0, cart_comm, &status);

    if (rank == 0) printf("[%d] Предварительный сдвиг завершён\n", rank);

    MPI_Barrier(cart_comm);

    start_time = MPI_Wtime();

    for (int step = 0; step < grid_size; step++) {
        block_multiply(A_block, B_block, C_block, block_size);
        if (DEBUG && step == 0) {
            printf("[%d] Итерация %d: выполнено локальное умножение блоков\n", rank, step);
        }
        MPI_Cart_shift(cart_comm, 1, -1, &src_rank, &dst_rank);
        MPI_Sendrecv_replace(A_block, block_size * block_size, MPI_DOUBLE,
                             dst_rank, 0, src_rank, 0, cart_comm, &status);

        MPI_Cart_shift(cart_comm, 0, -1, &src_rank, &dst_rank);
        MPI_Sendrecv_replace(B_block, block_size * block_size, MPI_DOUBLE,
                             dst_rank, 0, src_rank, 0, cart_comm, &status);
    }

    end_time = MPI_Wtime();
    parallel_time = end_time - start_time;

    if (rank == 0) printf("[%d] Основной цикл завершён\n", rank);

    if (rank == 0) printf("[%d] Сбор блоков результата...\n", rank);
    MPI_Gatherv(C_block, block_size * block_size, MPI_DOUBLE,
                 C_full, sendcounts, displs, resized_block,
                 0, cart_comm);
    if (rank == 0) printf("[%d] Блоки собраны\n", rank);

    if (rank == 0) {
        
        print_matrix(C_full, MATRIX_SIZE, "Результат C (параллельно)");
        double *C_seq = (double*)calloc(MATRIX_SIZE * MATRIX_SIZE, sizeof(double));
        start_time = MPI_Wtime();
        sequential_multiply(A_full, B_full, C_seq, MATRIX_SIZE);
        end_time = MPI_Wtime();
        seq_time = end_time - start_time;
        double speedup = seq_time / parallel_time;
        double efficiency = (speedup / size) * 100.0;

        double max_error = verify_result(C_full, C_seq, MATRIX_SIZE);
       
        printf("\n==========================\n");
        printf("Результаты\n");
        printf("============================\n");
        printf("Размер матриц:                      %d × %d\n", MATRIX_SIZE, MATRIX_SIZE);
        printf("Количество процессов:               %d\n", size);
        printf("Время параллельного выполнения:     %.6f сек\n", parallel_time);
        printf("Время последовательного выполнения: %.6f сек\n", seq_time);
        printf("Ускорение:                          %.2f раз\n", speedup);
        printf("Эффективность параллелизма:         %.2f%%\n", efficiency);
        printf(" \n\n");

        free(C_seq);
        free(A_full);
        free(B_full);
        free(C_full);
        free(sendcounts);
        free(displs);
    }


    free(A_block);
    free(B_block);
    free(C_block);
    MPI_Type_free(&resized_block);
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();

    return 0;
}
