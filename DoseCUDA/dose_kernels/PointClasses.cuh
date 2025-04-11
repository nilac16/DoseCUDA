#ifndef POINT_CLASSES_H
#define POINT_CLASSES_H

#include <cstddef>

typedef struct {

    unsigned i;
    unsigned j;
    unsigned k;

} PointIJK;

typedef struct {

    float x;
    float y;
    float z;

} PointXYZ;

__host__ __device__ static inline float xyz_dotproduct(const PointXYZ &a, const PointXYZ &b) {

    return a.x * b.x + a.y * b.y + a.z * b.z;
}

#endif /* POINT_CLASSES_H */
