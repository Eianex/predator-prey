#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

static int get_rw_buffer(PyObject *obj, Py_buffer *view) {
    return PyObject_GetBuffer(obj, view, PyBUF_WRITABLE | PyBUF_ND | PyBUF_FORMAT | PyBUF_C_CONTIGUOUS);
}

static int get_ro_buffer(PyObject *obj, Py_buffer *view) {
    return PyObject_GetBuffer(obj, view, PyBUF_ND | PyBUF_FORMAT | PyBUF_C_CONTIGUOUS);
}

static int check_1d_len(Py_buffer *view, Py_ssize_t itemsize, const char *name) {
    if (view->ndim != 1) {
        PyErr_Format(PyExc_ValueError, "%s must be 1D", name);
        return 0;
    }
    if (view->itemsize != itemsize) {
        PyErr_Format(PyExc_ValueError, "%s has wrong itemsize", name);
        return 0;
    }
    return 1;
}

static int clamp_and_bounce(float *px, float *py, float *vx, float *vy,
                            float radius, float width, float height) {
    int changed = 0;
    if (*px < radius) {
        *px = radius;
        *vx = -*vx;
        changed = 1;
    } else if (*px > width - radius) {
        *px = width - radius;
        *vx = -*vx;
        changed = 1;
    }

    if (*py < radius) {
        *py = radius;
        *vy = -*vy;
        changed = 1;
    } else if (*py > height - radius) {
        *py = height - radius;
        *vy = -*vy;
        changed = 1;
    }
    return changed;
}

static PyObject *py_solve_collisions_same_species_raw(PyObject *self, PyObject *args) {
    PyObject *x_obj, *y_obj, *vx_obj, *vy_obj, *radius_obj, *speed_obj, *alive_obj;
    double width_d, height_d;
    if (!PyArg_ParseTuple(
            args,
            "OOOOOOOdd",
            &x_obj, &y_obj, &vx_obj, &vy_obj, &radius_obj, &speed_obj, &alive_obj,
            &width_d, &height_d)) {
        return NULL;
    }

    Py_buffer x = {0}, y = {0}, vx = {0}, vy = {0}, radius = {0}, speed = {0}, alive = {0};
    PyObject *touched = NULL;

    if (get_rw_buffer(x_obj, &x) < 0 || get_rw_buffer(y_obj, &y) < 0 ||
        get_rw_buffer(vx_obj, &vx) < 0 || get_rw_buffer(vy_obj, &vy) < 0 ||
        get_ro_buffer(radius_obj, &radius) < 0 || get_ro_buffer(speed_obj, &speed) < 0 ||
        get_ro_buffer(alive_obj, &alive) < 0) {
        goto fail;
    }

    if (!check_1d_len(&x, sizeof(float), "x") || !check_1d_len(&y, sizeof(float), "y") ||
        !check_1d_len(&vx, sizeof(float), "vx") || !check_1d_len(&vy, sizeof(float), "vy") ||
        !check_1d_len(&radius, sizeof(float), "radius") || !check_1d_len(&speed, sizeof(float), "speed") ||
        !check_1d_len(&alive, sizeof(uint8_t), "alive")) {
        goto fail;
    }

    Py_ssize_t n = x.len / (Py_ssize_t)sizeof(float);
    if ((y.len / (Py_ssize_t)sizeof(float)) != n || (vx.len / (Py_ssize_t)sizeof(float)) != n ||
        (vy.len / (Py_ssize_t)sizeof(float)) != n || (radius.len / (Py_ssize_t)sizeof(float)) != n ||
        (speed.len / (Py_ssize_t)sizeof(float)) != n || (alive.len / (Py_ssize_t)sizeof(uint8_t)) != n) {
        PyErr_SetString(PyExc_ValueError, "all arrays must have the same length");
        goto fail;
    }

    touched = PyByteArray_FromStringAndSize(NULL, n);
    if (!touched) {
        goto fail;
    }
    memset(PyByteArray_AS_STRING(touched), 0, (size_t)n);

    float *xp = (float *)x.buf;
    float *yp = (float *)y.buf;
    float *vxp = (float *)vx.buf;
    float *vyp = (float *)vy.buf;
    const float *rp = (const float *)radius.buf;
    const float *sp = (const float *)speed.buf;
    const uint8_t *ap = (const uint8_t *)alive.buf;
    uint8_t *tp = (uint8_t *)PyByteArray_AS_STRING(touched);
    const float width = (float)width_d;
    const float height = (float)height_d;

    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < n - 1; ++i) {
        if (!ap[i]) {
            continue;
        }

        float xi = xp[i];
        float yi = yp[i];
        float vxi = vxp[i];
        float vyi = vyp[i];
        const float ri = rp[i];
        const float si = sp[i];

        for (Py_ssize_t j = i + 1; j < n; ++j) {
            if (!ap[j]) {
                continue;
            }

            float xj = xp[j];
            float yj = yp[j];
            float vxj = vxp[j];
            float vyj = vyp[j];

            const float dx = xj - xi;
            const float dy = yj - yi;
            const float min_dist = ri + rp[j];
            const float dist_sq = dx * dx + dy * dy;

            if (dist_sq >= min_dist * min_dist) {
                continue;
            }

            float nx, ny, dist;
            if (dist_sq < 1e-12f) {
                nx = 1.0f;
                ny = 0.0f;
                dist = 0.0f;
            } else {
                dist = sqrtf(dist_sq);
                const float inv = 1.0f / dist;
                nx = dx * inv;
                ny = dy * inv;
            }

            const float overlap = min_dist - dist;
            if (overlap > 0.0f) {
                const float half = 0.5f * overlap;
                const float corrx = nx * half;
                const float corry = ny * half;
                xi -= corrx;
                yi -= corry;
                xj += corrx;
                yj += corry;
            }

            clamp_and_bounce(&xi, &yi, &vxi, &vyi, ri, width, height);
            clamp_and_bounce(&xj, &yj, &vxj, &vyj, rp[j], width, height);

            const float va_n = vxi * nx + vyi * ny;
            const float vb_n = vxj * nx + vyj * ny;

            if (va_n - vb_n > 0.0f) {
                const float delta_a = vb_n - va_n;
                const float delta_b = va_n - vb_n;

                vxi += delta_a * nx;
                vyi += delta_a * ny;
                vxj += delta_b * nx;
                vyj += delta_b * ny;

                const float la2 = vxi * vxi + vyi * vyi;
                if (la2 > 1e-12f) {
                    const float scale_a = si / sqrtf(la2);
                    vxi *= scale_a;
                    vyi *= scale_a;
                }

                const float sj = sp[j];
                const float lb2 = vxj * vxj + vyj * vyj;
                if (lb2 > 1e-12f) {
                    const float scale_b = sj / sqrtf(lb2);
                    vxj *= scale_b;
                    vyj *= scale_b;
                }
            }

            xp[j] = xj;
            yp[j] = yj;
            vxp[j] = vxj;
            vyp[j] = vyj;

            tp[i] = 1;
            tp[j] = 1;
        }

        xp[i] = xi;
        yp[i] = yi;
        vxp[i] = vxi;
        vyp[i] = vyi;
    }
    Py_END_ALLOW_THREADS

    PyBuffer_Release(&x);
    PyBuffer_Release(&y);
    PyBuffer_Release(&vx);
    PyBuffer_Release(&vy);
    PyBuffer_Release(&radius);
    PyBuffer_Release(&speed);
    PyBuffer_Release(&alive);
    return touched;

fail:
    if (touched) Py_DECREF(touched);
    if (x.buf) PyBuffer_Release(&x);
    if (y.buf) PyBuffer_Release(&y);
    if (vx.buf) PyBuffer_Release(&vx);
    if (vy.buf) PyBuffer_Release(&vy);
    if (radius.buf) PyBuffer_Release(&radius);
    if (speed.buf) PyBuffer_Release(&speed);
    if (alive.buf) PyBuffer_Release(&alive);
    return NULL;
}

static PyObject *py_collect_nearby_indices(PyObject *self, PyObject *args) {
    PyObject *x_obj, *y_obj, *ids_obj, *alive_obj, *out_idx_obj;
    double qx, qy, radius_sq;
    long long exclude_id;
    if (!PyArg_ParseTuple(args, "OOOOdddLO", &x_obj, &y_obj, &ids_obj, &alive_obj,
                          &qx, &qy, &radius_sq, &exclude_id, &out_idx_obj)) {
        return NULL;
    }

    Py_buffer x = {0}, y = {0}, ids = {0}, alive = {0}, out_idx = {0};
    if (get_ro_buffer(x_obj, &x) < 0 || get_ro_buffer(y_obj, &y) < 0 ||
        get_ro_buffer(ids_obj, &ids) < 0 || get_ro_buffer(alive_obj, &alive) < 0 ||
        get_rw_buffer(out_idx_obj, &out_idx) < 0) {
        goto fail;
    }

    if (!check_1d_len(&x, sizeof(float), "x") || !check_1d_len(&y, sizeof(float), "y") ||
        !check_1d_len(&ids, sizeof(int64_t), "ids") || !check_1d_len(&alive, sizeof(uint8_t), "alive") ||
        !check_1d_len(&out_idx, sizeof(int32_t), "out_idx")) {
        goto fail;
    }

    Py_ssize_t n = x.len / (Py_ssize_t)sizeof(float);
    if ((y.len / (Py_ssize_t)sizeof(float)) != n || (ids.len / (Py_ssize_t)sizeof(int64_t)) != n ||
        (alive.len / (Py_ssize_t)sizeof(uint8_t)) != n || (out_idx.len / (Py_ssize_t)sizeof(int32_t)) < n) {
        PyErr_SetString(PyExc_ValueError, "bad array lengths");
        goto fail;
    }

    const float *xp = (const float *)x.buf;
    const float *yp = (const float *)y.buf;
    const int64_t *idp = (const int64_t *)ids.buf;
    const uint8_t *ap = (const uint8_t *)alive.buf;
    int32_t *op = (int32_t *)out_idx.buf;
    int32_t count = 0;
    const float qxf = (float)qx;
    const float qyf = (float)qy;
    const float rsq = (float)radius_sq;
    const int64_t ex = (int64_t)exclude_id;

    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < n; ++i) {
        if (!ap[i]) continue;
        if (ex >= 0 && idp[i] == ex) continue;
        const float dx = xp[i] - qxf;
        const float dy = yp[i] - qyf;
        if (dx * dx + dy * dy <= rsq) {
            op[count++] = (int32_t)i;
        }
    }
    Py_END_ALLOW_THREADS

    PyBuffer_Release(&x);
    PyBuffer_Release(&y);
    PyBuffer_Release(&ids);
    PyBuffer_Release(&alive);
    PyBuffer_Release(&out_idx);
    return PyLong_FromLong((long)count);

fail:
    if (x.buf) PyBuffer_Release(&x);
    if (y.buf) PyBuffer_Release(&y);
    if (ids.buf) PyBuffer_Release(&ids);
    if (alive.buf) PyBuffer_Release(&alive);
    if (out_idx.buf) PyBuffer_Release(&out_idx);
    return NULL;
}

static PyObject *py_nearest_alive_index(PyObject *self, PyObject *args) {
    PyObject *x_obj, *y_obj, *ids_obj, *alive_obj;
    double qx, qy;
    long long exclude_id;
    if (!PyArg_ParseTuple(args, "OOOOddL", &x_obj, &y_obj, &ids_obj, &alive_obj, &qx, &qy, &exclude_id)) {
        return NULL;
    }

    Py_buffer x = {0}, y = {0}, ids = {0}, alive = {0};
    if (get_ro_buffer(x_obj, &x) < 0 || get_ro_buffer(y_obj, &y) < 0 ||
        get_ro_buffer(ids_obj, &ids) < 0 || get_ro_buffer(alive_obj, &alive) < 0) {
        goto fail;
    }

    if (!check_1d_len(&x, sizeof(float), "x") || !check_1d_len(&y, sizeof(float), "y") ||
        !check_1d_len(&ids, sizeof(int64_t), "ids") || !check_1d_len(&alive, sizeof(uint8_t), "alive")) {
        goto fail;
    }

    Py_ssize_t n = x.len / (Py_ssize_t)sizeof(float);
    if ((y.len / (Py_ssize_t)sizeof(float)) != n || (ids.len / (Py_ssize_t)sizeof(int64_t)) != n ||
        (alive.len / (Py_ssize_t)sizeof(uint8_t)) != n) {
        PyErr_SetString(PyExc_ValueError, "bad array lengths");
        goto fail;
    }

    const float *xp = (const float *)x.buf;
    const float *yp = (const float *)y.buf;
    const int64_t *idp = (const int64_t *)ids.buf;
    const uint8_t *ap = (const uint8_t *)alive.buf;
    const float qxf = (float)qx;
    const float qyf = (float)qy;
    const int64_t ex = (int64_t)exclude_id;
    int32_t best = -1;
    float best_d2 = 1e30f;

    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < n; ++i) {
        if (!ap[i]) continue;
        if (ex >= 0 && idp[i] == ex) continue;
        const float dx = xp[i] - qxf;
        const float dy = yp[i] - qyf;
        const float d2 = dx * dx + dy * dy;
        if (d2 < best_d2) {
            best_d2 = d2;
            best = (int32_t)i;
        }
    }
    Py_END_ALLOW_THREADS

    PyBuffer_Release(&x);
    PyBuffer_Release(&y);
    PyBuffer_Release(&ids);
    PyBuffer_Release(&alive);
    return PyLong_FromLong((long)best);

fail:
    if (x.buf) PyBuffer_Release(&x);
    if (y.buf) PyBuffer_Release(&y);
    if (ids.buf) PyBuffer_Release(&ids);
    if (alive.buf) PyBuffer_Release(&alive);
    return NULL;
}

static PyObject *py_nearest_alive_grown_index(PyObject *self, PyObject *args) {
    PyObject *x_obj, *y_obj, *ids_obj, *alive_obj, *grown_obj;
    double qx, qy;
    long long exclude_id;
    if (!PyArg_ParseTuple(args, "OOOOOddL", &x_obj, &y_obj, &ids_obj, &alive_obj, &grown_obj, &qx, &qy, &exclude_id)) {
        return NULL;
    }

    Py_buffer x = {0}, y = {0}, ids = {0}, alive = {0}, grown = {0};
    if (get_ro_buffer(x_obj, &x) < 0 || get_ro_buffer(y_obj, &y) < 0 ||
        get_ro_buffer(ids_obj, &ids) < 0 || get_ro_buffer(alive_obj, &alive) < 0 ||
        get_ro_buffer(grown_obj, &grown) < 0) {
        goto fail;
    }

    if (!check_1d_len(&x, sizeof(float), "x") || !check_1d_len(&y, sizeof(float), "y") ||
        !check_1d_len(&ids, sizeof(int64_t), "ids") || !check_1d_len(&alive, sizeof(uint8_t), "alive") ||
        !check_1d_len(&grown, sizeof(uint8_t), "grown")) {
        goto fail;
    }

    Py_ssize_t n = x.len / (Py_ssize_t)sizeof(float);
    if ((y.len / (Py_ssize_t)sizeof(float)) != n || (ids.len / (Py_ssize_t)sizeof(int64_t)) != n ||
        (alive.len / (Py_ssize_t)sizeof(uint8_t)) != n || (grown.len / (Py_ssize_t)sizeof(uint8_t)) != n) {
        PyErr_SetString(PyExc_ValueError, "bad array lengths");
        goto fail;
    }

    const float *xp = (const float *)x.buf;
    const float *yp = (const float *)y.buf;
    const int64_t *idp = (const int64_t *)ids.buf;
    const uint8_t *ap = (const uint8_t *)alive.buf;
    const uint8_t *gp = (const uint8_t *)grown.buf;
    const float qxf = (float)qx;
    const float qyf = (float)qy;
    const int64_t ex = (int64_t)exclude_id;
    int32_t best = -1;
    float best_d2 = 1e30f;

    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < n; ++i) {
        if (!ap[i]) continue;
        if (!gp[i]) continue;
        if (ex >= 0 && idp[i] == ex) continue;
        const float dx = xp[i] - qxf;
        const float dy = yp[i] - qyf;
        const float d2 = dx * dx + dy * dy;
        if (d2 < best_d2) {
            best_d2 = d2;
            best = (int32_t)i;
        }
    }
    Py_END_ALLOW_THREADS

    PyBuffer_Release(&x);
    PyBuffer_Release(&y);
    PyBuffer_Release(&ids);
    PyBuffer_Release(&alive);
    PyBuffer_Release(&grown);
    return PyLong_FromLong((long)best);

fail:
    if (x.buf) PyBuffer_Release(&x);
    if (y.buf) PyBuffer_Release(&y);
    if (ids.buf) PyBuffer_Release(&ids);
    if (alive.buf) PyBuffer_Release(&alive);
    if (grown.buf) PyBuffer_Release(&grown);
    return NULL;
}

static PyMethodDef FastKernelsMethods[] = {
    {"solve_collisions_same_species_raw", py_solve_collisions_same_species_raw, METH_VARARGS, "Collision solver over contiguous float32/uint8 arrays."},
    {"collect_nearby_indices", py_collect_nearby_indices, METH_VARARGS, "Fill out_idx with nearby indices and return count."},
    {"nearest_alive_index", py_nearest_alive_index, METH_VARARGS, "Return nearest alive index or -1."},
    {"nearest_alive_grown_index", py_nearest_alive_grown_index, METH_VARARGS, "Return nearest alive+grown index or -1."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef fastkernelsmodule = {
    PyModuleDef_HEAD_INIT,
    "_fastkernels",
    "Fast kernels for collisions and proximity queries.",
    -1,
    FastKernelsMethods
};

PyMODINIT_FUNC PyInit__fastkernels(void) {
    return PyModule_Create(&fastkernelsmodule);
}
