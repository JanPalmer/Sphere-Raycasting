//#ifndef HITTABLE_LIST_H
//#define HITTABLE_LIST_H
//
//#include "hittable.cuh"
//
//#include <memory>
//#include <vector>
//
//using std::shared_ptr;
//using std::make_shared;
//
//class hittable_list : public hittable {
//public:
//	__host__ __device__ hittable_list() {}
//	__host__ __device__ hittable_list(shared_ptr<hittable> object) { add(object); }
//
//	__host__ __device__ void clear() { objects.clear(); }
//	__host__ __device__ void add(shared_ptr<hittable> object) { objects.push_back(object); }
//
//	__host__ __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;
//
//public:
//	std::vector<shared_ptr<hittable>> objects;
//};
//
//__host__ __device__ inline bool hittable_list::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
//	hit_record temp_rec;
//	bool hit_anything = false;
//	auto closest_so_far = t_max;
//
//	for (const auto& object : objects) {
//		if (object->hit(r, t_min, closest_so_far, temp_rec)) {
//			hit_anything = true;
//			closest_so_far = temp_rec.t;
//			rec = temp_rec;
//		}
//	}
//
//	return hit_anything;
//}
//#endif