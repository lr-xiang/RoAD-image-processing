#include <iostream>
#include <fstream>
#include <string>
#include <math.h> 
#include <exception>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/core.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/point_types_conversion.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <boost/thread/thread.hpp>
#include <boost/filesystem.hpp> 

#include <pcl/registration/icp.h>
#include <pcl/registration/joint_icp.h>
#include <pcl/registration/transformation_estimation_point_to_plane.h>
#include <pcl/registration/correspondence_rejection_median_distance.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/console/time.h>   // TicToc

#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/features/normal_3d.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/radius_outlier_removal.h>

#include <pcl/segmentation/extract_clusters.h>  //euclidean cluster
#include <pcl/segmentation/region_growing_rgb.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/segmentation/lccp_segmentation.h>
#include <pcl/segmentation/cpc_segmentation.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/segmentation/min_cut_segmentation.h>

#include <pcl/surface/mls.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/surface/concave_hull.h>
#include <pcl/surface/gp3.h>

#include <pcl/features/organized_edge_detection.h>
#include <pcl/features/normal_3d.h>

#include <Eigen/Dense>  //using vector
#include <vector>

#include <pcl/ModelCoefficients.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>

#include <vtkPolyLine.h>

#include <pcl/common/pca.h>

#include <utility>

#include <pcl/filters/passthrough.h>

#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/bc_clustering.hpp>
#include <boost/graph/iteration_macros.hpp>
#include <boost/graph/copy.hpp>
#include <boost/algorithm/string/replace.hpp>
#include <boost/filesystem.hpp>

#include <boost/algorithm/string.hpp>

#include <boost/config.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/graph/two_bit_color_map.hpp>
#include <boost/graph/named_function_params.hpp>
#include <boost/graph/properties.hpp>
#include <boost/graph/iteration_macros.hpp>
#include <boost/graph/kruskal_min_spanning_tree.hpp>
#include <boost/graph/prim_minimum_spanning_tree.hpp>
#include <boost/graph/connected_components.hpp>

#include <stdlib.h>     /* srand, rand */
#include <time.h>   
#include <stdio.h>  


using namespace boost;

//#define visualize
#define leafsegmentation  
#define PI 3.14159265

int exp_NO = 19;
int to_merge_th = 2000;

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::PointXYZL PointLT;
typedef pcl::PointCloud<PointLT> PointLCloudT;

typedef pcl::SupervoxelClustering<PointT>::VoxelAdjacencyList Graph;
typedef boost::graph_traits<Graph>::vertex_iterator supervoxel_iterator;
typedef boost::graph_traits<Graph>::edge_iterator supervoxel_edge_iter;
typedef pcl::SupervoxelClustering<PointT>::VoxelID Voxel;
typedef pcl::SupervoxelClustering<PointT>::EdgeID Edge;
typedef pcl::SupervoxelClustering<PointT>::VoxelAdjacencyList::adjacency_iterator supervoxel_adjacency_iterator;

struct SVEdgeProperty
{
	float weight;
};

struct SVVertexProperty
{
	uint32_t supervoxel_label;
	pcl::Supervoxel<PointT>::Ptr supervoxel;
	uint32_t index;
	float max_width;
	int vertex;
	bool near_junction = false;
	std::vector<uint32_t> children;
	std::vector<uint32_t> parents;
	std::vector<int> cluster_indices;
};


typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS, SVVertexProperty, SVEdgeProperty> sv_graph_t;

typedef boost::graph_traits<sv_graph_t>::vertex_descriptor sv_vertex_t;

typedef boost::graph_traits<sv_graph_t>::edge_descriptor sv_edge_t;

cv::Mat kinect_rgb_camera_matrix_cv_, kinect_rgb_dist_coeffs_cv_;
cv::Mat image;
cv::Mat rgb_pose_cv_0_, rgb_pose_cv_1_, rgb_pose_cv_2_, rgb_pose_cv_3_, rgb_pose_cv_4_;
Eigen::Matrix4d rgb_pose_0, rgb_pose_1, rgb_pose_2, rgb_pose_3, rgb_pose_4;

PointT pot_center;

using namespace std;

using namespace pcl;
using namespace pcl::io;

bool leaf_file_first_line = true;
bool measure_file_first_line = true;

string current_pot_id, current_date;

struct Smooth {

	std::vector<int> cluster_indices;

	int cluster_size;

	float distance_to_center;

	float z_value;

	float scaled_cluster_th;

	bool checked = false;

};


struct Leaf {

	int num_size = -1;

	float leaf_angle = -1;

	double leaf_curvature = -1;

	float leaf_smoothness = -1;

	float leaf_length = -1;

	float leaf_width = -1;

	float aspect_ratio = -1; //length/width

	float leaf_area = -1;

	float leaf_convexhull_area = -1;

	float leaf_convexity = -1;

	float leaf_boundingbox_volume = -1;

	float leaf_convexhull_volume = -1;

	float leaf_solidity = -1;

};

bool scompare(const Smooth &l1, const Smooth &l2) { return l1.cluster_size > l2.cluster_size; }
bool zcompare(const Smooth &l1, const Smooth &l2) { return l1.z_value < l2.z_value; }
bool cloud_size_compare(const PointCloudT::Ptr &p1, const PointCloudT::Ptr &p2) { return p1->size()>p2->size(); }

float pre_x = 0, pre_y = 0, pre_z = 0, Dist = 0;
void pp_callback(const pcl::visualization::PointPickingEvent& event);

int loadData(std::string& id, std::string& date,
	std::vector<PointCloudT::Ptr, Eigen::aligned_allocator <PointCloudT::Ptr>>& data);


int ICPregistrationMatrix(std::vector<PointCloudT::Ptr, Eigen::aligned_allocator<PointCloudT::Ptr> > &data,
	std::vector<Eigen::Matrix4d>& matrixVector, PointCloudT::Ptr cloud_out);

inline void removePotOutlier(PointCloudT::Ptr cloud_in,
	PointCloudT::Ptr cloud_out);

void
print4x4Matrix(const Eigen::Matrix4d & matrix);

int regionGrowing(PointCloudT::Ptr cloud, PointCloudT::Ptr cloud_out);

void registerRGBandPointCloud(cv::Mat & rgb, PointCloudT::Ptr scan_cloud, Eigen::Matrix4d & rgb_pose);

inline void downsample(PointCloudT::Ptr cloud, PointCloudT::Ptr cloud_out, float leaf_size);

float getMeshArea(PointCloudT::Ptr cloud);

float getMicroCurvature(PointCloudT::Ptr cloud);

inline int getEuclideanSize(PointCloudT::Ptr cloud);

void leafPhenotype(vector<PointCloudT::Ptr> &vec);

void leafSeg2(PointCloudT::Ptr cloud, vector <pcl::PointIndices> &clusters);

string current_exp_folder, leaf_measurement_path;

//usage: ./merge /media/lietang/easystore1/RoAD/exp15/ 15

int
main(int argc,
	char* argv[])
{

	current_exp_folder = argv[1];
	
	exp_NO = stoi(argv[2]);

	leaf_measurement_path = current_exp_folder + "leaf_seg_eigen_dis2/individual_leaf_traits.csv";

	boost::filesystem::path dir_3d_images(current_exp_folder + "3d_images");

	if (boost::filesystem::create_directory(dir_3d_images))
	{
		std::cerr << "Directory Created for 3d images\n";
	}


	boost::filesystem::path dir_3d_mesh(current_exp_folder + "3d_mesh");

	if (boost::filesystem::create_directory(dir_3d_mesh))
	{
		std::cerr << "Directory Created for 3d mesh \n";
	}

	boost::filesystem::path dir_3d_leaf(current_exp_folder + "leaf_seg_eigen_dis2");

	if (boost::filesystem::create_directory(dir_3d_leaf))
	{
		std::cerr << "Directory Created for 3d leaf\n";
	}


	cv::FileStorage fs("kinectRGBCalibration.yml", cv::FileStorage::READ);

	if (fs.isOpened())
	{
		fs["camera_matrix"] >> kinect_rgb_camera_matrix_cv_;

		fs["distortion_coefficients"] >> kinect_rgb_dist_coeffs_cv_;

		fs.release();
	}
	
				
	string rgb_pose_file = "rgb_pose.yml";
	if(exp_NO ==6)
		rgb_pose_file = "rgb_pose_exp6.yml";	
	if(exp_NO ==10)
		rgb_pose_file = "rgb_pose_exp10.yml";
	if(exp_NO ==13)
		rgb_pose_file = "rgb_pose_exp13.yml";
	if(exp_NO >=16)
		rgb_pose_file = "rgb_pose_exp18.yml";
				
	cv::FileStorage fs_pose(rgb_pose_file, cv::FileStorage::READ);
	if (fs_pose.isOpened())
	{
		fs_pose["rgb_pose_0"] >> rgb_pose_cv_0_;
		fs_pose["rgb_pose_1"] >> rgb_pose_cv_1_;
		fs_pose["rgb_pose_2"] >> rgb_pose_cv_2_;
		fs_pose["rgb_pose_3"] >> rgb_pose_cv_3_;
		fs_pose["rgb_pose_4"] >> rgb_pose_cv_4_;

		fs_pose.release();
	}
	
	cout<<"rgb_pose_cv_0_: "<<rgb_pose_cv_0_<<endl;
	cout<<"rgb_pose_cv_1_: "<<rgb_pose_cv_1_<<endl;
	cout<<"rgb_pose_cv_2_: "<<rgb_pose_cv_2_<<endl;
	cout<<"rgb_pose_cv_3_: "<<rgb_pose_cv_3_<<endl;
	cout<<"rgb_pose_cv_4_: "<<rgb_pose_cv_4_<<endl;
	
	for (int y = 0; y < 4; y++) {
		for (int x = 0; x < 4; x++) {
			rgb_pose_0(y, x) = rgb_pose_cv_0_.at<double>(y, x);
			rgb_pose_1(y, x) = rgb_pose_cv_1_.at<double>(y, x);
			rgb_pose_2(y, x) = rgb_pose_cv_2_.at<double>(y, x);
			rgb_pose_3(y, x) = rgb_pose_cv_3_.at<double>(y, x);
			rgb_pose_4(y, x) = rgb_pose_cv_4_.at<double>(y, x);
		}
	}
	
	cout<<"rgb_pose_0: "<<rgb_pose_0<<endl;
	cout<<"rgb_pose_1: "<<rgb_pose_1<<endl;
	cout<<"rgb_pose_2: "<<rgb_pose_2<<endl;
	cout<<"rgb_pose_3: "<<rgb_pose_3<<endl;
	cout<<"rgb_pose_4: "<<rgb_pose_4<<endl;		
			

	int start_i=0;
	int start_j=0;

	cout << "start_i: " << start_i << ", start_j: " << start_j << endl;

	std::vector<std::string> pot_labels;
	std::vector<std::string> date;


	std::ifstream label_input(current_exp_folder + "experiment_label.txt");  
	std::ifstream date_input(current_exp_folder + "experiment_date.txt"); 

	if (label_input.is_open())
	{
		int line_num = 0;
		for (std::string line; std::getline(label_input, line); line_num++){
			boost::replace_all(line, "\r","");
			boost::replace_all(line, "\n","");
			pot_labels.push_back(line);
			//cout<<"line: "<<line<<endl;
		}
			
	}
	if (date_input.is_open())
	{
		int line_num = 0;
		for (std::string line; std::getline(date_input, line); line_num++){
			boost::replace_all(line, "\r","");
			boost::replace_all(line, "\n","");
			date.push_back(line);
			//cout<<"date: "<<line<<endl;
		}
			
	}

	std::cout << "pot size: " << pot_labels.size() << " date size: " << date.size() << std::endl;

	for (int i = 0;i < pot_labels.size();i++) {
		for (int j = 0;j < date.size();j++) {

			current_pot_id = pot_labels[i];
			current_date = date[j];
			cout<<"i: "<<i<<" j: "<<j<<endl;	

			std::vector<PointCloudT::Ptr, Eigen::aligned_allocator<PointCloudT::Ptr> > data;

			if (i == start_i && j < start_j) continue;

			std::vector<Eigen::Matrix4d> matrixVector;

			PointCloudT::Ptr cloud_merged(new PointCloudT);


			if (loadData(pot_labels[i], date[j], data) < 0)
				continue;


			if (ICPregistrationMatrix(data, matrixVector, cloud_merged) < 0)
				continue;


//save data
#ifndef visualize
			
			std::string CurrentOutputFolder = "";

			CurrentOutputFolder = current_exp_folder+"3d_images/" + pot_labels[i];

			boost::filesystem::path dir(CurrentOutputFolder.c_str());

			if (boost::filesystem::create_directory(dir))
			{
				std::cerr << "Directory Created: " << CurrentOutputFolder << std::endl;
			}


			std::stringstream file_name;
			file_name << CurrentOutputFolder.c_str() << "/" << pot_labels[i] << "_" << date[j] << ".pcd";

	//save pcd file
			pcl::io::savePCDFile(file_name.str(), *cloud_merged);
#endif

		}

	}

	cout << "All finished!!\n";
	getchar();
	return 0;
}

void pp_callback(const pcl::visualization::PointPickingEvent& event)
{
	float x, y, z;
	event.getPoint(x, y, z);
	Dist = sqrt(pow(x - pre_x, 2) + pow(y - pre_y, 2) + pow(z - pre_z, 2));
	//	Eigen::Vector3f dir(pre_x-x, pre_y-y, pre_z-z);
	//	dir.normalize();	
	pre_x = x;
	pre_y = y;
	pre_z = z;
	std::cout << "x:" << x << " y:" << y << " z:" << z << " distance:" << Dist/*<<" nx:"<<dir(0)<<" ny:"<<dir(1)<<" nz:"<<dir(2)*/ << std::endl;

}

void
print4x4Matrix(const Eigen::Matrix4d & matrix)
{
	printf("Rotation matrix :\n");
	printf("    | %6.3f %6.3f %6.3f | \n", matrix(0, 0), matrix(0, 1), matrix(0, 2));
	printf("R = | %6.3f %6.3f %6.3f | \n", matrix(1, 0), matrix(1, 1), matrix(1, 2));
	printf("    | %6.3f %6.3f %6.3f | \n", matrix(2, 0), matrix(2, 1), matrix(2, 2));
	printf("Translation vector :\n");
	printf("t = < %6.3f, %6.3f, %6.3f >\n\n", matrix(0, 3), matrix(1, 3), matrix(2, 3));
}

int loadData(std::string& id, std::string& date,
	std::vector<PointCloudT::Ptr, Eigen::aligned_allocator <PointCloudT::Ptr>>& data) {

	Eigen::Matrix4d rgb_pose;

	cout << "loading " << id << " at " << date << endl;

	PointCloudT::Ptr cloud(new PointCloudT);

	std::string path = current_exp_folder + date + "/" + id + "/" + id;

	std::string temp_path = path + "_scan_0.pcd";

	//cout<<temp_path <<endl;

	if (pcl::io::loadPCDFile<PointT>(temp_path, *cloud) == -1) //* load the file
	{

		PCL_ERROR("Couldn't read file test_pcd.pcd \n");
		return -1;
	}

	std::string rgbpath = current_exp_folder + "2d_images/" + id + "/processed/" + id + "_" + date + "_plant.bmp";

	if(exp_NO == 3 || exp_NO == 2)
		rgbpath = current_exp_folder + "2d_images/" + id + "/processed/" + id + "_" + date + "_plant3rd.bmp";

	if(exp_NO >= 15)
		rgbpath = current_exp_folder + "2d_images_deeplab/" + id + "/processed/" + id + "_" + date + "_plant.png";
		
	image = cv::imread(rgbpath, CV_LOAD_IMAGE_COLOR);
	if (image.empty())                      // Check for invalid input
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	cout<<cloud->size()<<endl;
	rgb_pose = rgb_pose_0;
	registerRGBandPointCloud(image, cloud, rgb_pose);
	data.push_back(cloud);

	cout << "data 0 : " << data.at(0)->size() << endl;

	return 1;
}

int ICPregistrationMatrix(std::vector<PointCloudT::Ptr, Eigen::aligned_allocator<PointCloudT::Ptr> > &data,
	std::vector<Eigen::Matrix4d>& matrixVector, PointCloudT::Ptr cloud_out) {

	cout << "ready for registration\n" << endl;

	PointCloudT::Ptr cloud_target(new PointCloudT);  // 

	PointCloudT::Ptr cloud_icp(new PointCloudT);  // ICP output point cloud

	PointCloudT::Ptr cloud(new PointCloudT);

	PointCloudT::Ptr cloud_plant(new PointCloudT);

	Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity();

	*cloud_target += *data.at(0);

	cout << "cloud_target original cloud:  " << cloud_target->size() << endl;

	removePotOutlier(cloud_target, cloud_target);

	pcl::console::TicToc time;

	pcl::IterativeClosestPoint<PointT, PointT> icp;

	icp.setMaxCorrespondenceDistance(0.01); //0.01

	icp.setTransformationEpsilon(1e-15);

	icp.setEuclideanFitnessEpsilon(1e-15);

	icp.setMaximumIterations(100);//100

	cloud->clear();

	*cloud += *cloud_target;

	if (regionGrowing(cloud_target, cloud_plant) < 0)
		return -1;

	cout << "cloud_plant: " << cloud_plant->size() << endl;

	if (cloud_plant->size() > to_merge_th) {

		downsample(cloud_target, cloud, 0.0004);

		for (int i = 1;i <data.size();i++) {

			PointCloudT::Ptr cloud_source(new PointCloudT);
			PointCloudT::Ptr cloud_icp_plant(new PointCloudT);
			PointCloudT::Ptr cloud_source_downsample(new PointCloudT);

			cloud_icp->clear();

			*cloud_source += *data.at(i);

			cout << "target cloud: " << cloud->size() << " points." << endl;

			cout << "pcd file " << i << " (source original):" << cloud_source->size() << " points." << endl;

			removePotOutlier(cloud_source, cloud_source);

			cout << "source cloud: " << cloud_source->size() << " points." << endl;

			time.tic();

			downsample(cloud_source, cloud_source_downsample, 0.0004);

			icp.setInputTarget(cloud);

			icp.setInputSource(cloud_source_downsample);

			icp.align(*cloud_icp);

			*cloud += *cloud_icp;

			cout << "round " << i << " finished registration in " << time.toc() << " ms.\n" << endl;;

			transformation_matrix = icp.getFinalTransformation().cast<double>();

			//print4x4Matrix(transformation_matrix);

			//for visualize
#ifdef visualize
			pcl::visualization::PCLVisualizer viewer("ICP demo");

			int v1(0);
			int v2(1);
			viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
			viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);

			viewer.registerPointPickingCallback(&pp_callback);

			viewer.addPointCloud(cloud, to_string(cv::getTickCount()), v1);

			pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_target_color_h(cloud_icp, 20, 20, 180);
			viewer.addPointCloud(cloud_icp, cloud_target_color_h, to_string(cv::getTickCount()), v1);

			pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_source_color_h(cloud_source_downsample, 20, 20, 180);
			viewer.addPointCloud(cloud_source_downsample, cloud_source_color_h, to_string(cv::getTickCount()), v2);

			viewer.addPointCloud(cloud, to_string(cv::getTickCount()), v2);

			viewer.spin();
#endif
			regionGrowing(cloud_source, cloud_source);
			pcl::transformPointCloud(*cloud_source, *cloud_icp_plant, transformation_matrix);
			*cloud_plant += *cloud_icp_plant;

			cout << "cloud_plant: " << cloud_plant->size() << endl;

#ifdef visualize
			pcl::visualization::PCLVisualizer viewer0("ICP demo");

			int v10(0);
			int v20(1);
			viewer0.createViewPort(0.0, 0.0, 0.5, 1.0, v10);
			viewer0.createViewPort(0.5, 0.0, 1.0, 1.0, v20);

			viewer0.registerPointPickingCallback(&pp_callback);

			viewer0.addPointCloud(cloud_plant, to_string(cv::getTickCount()), v10);

			pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_target_color_h0(cloud_icp_plant, 20, 20, 180);
			viewer0.addPointCloud(cloud_icp_plant, cloud_target_color_h0, to_string(cv::getTickCount()), v10);

			pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_source_color_h0(cloud_source, 20, 20, 180);
			viewer0.addPointCloud(cloud_source, cloud_source_color_h0, to_string(cv::getTickCount()), v20);

			viewer0.addPointCloud(cloud_plant, to_string(cv::getTickCount()), v20);

			viewer0.spin();
#endif
		}
	}
	
	downsample(cloud_plant, cloud_out, 0.0004);

	cout << "registration done!\n" << endl;

	cout << "cloud plant final: " << cloud_out->size() << "\n" << endl;

	//getchar();

	return 1;
}

inline void removePotOutlier(PointCloudT::Ptr cloud_in,
	PointCloudT::Ptr cloud_out) {
	cout << "removePotOutlier\n";
	//remove outlies by cone

	PointCloudT::Ptr cloud_pot(new PointCloudT);

	PointCloudT::Ptr cloud_icp(new PointCloudT);

	PointCloudT::Ptr cloud_backup(new PointCloudT);

	*cloud_backup += *cloud_in;

	float center_x = -0.06f;

	float center_y = 0.43f;

	float center_z = 0.029f;

	float rad = 0.045;

	for (int i = 0;i < 360;++i) {

		float s = sin(i*3.1415926 / 180);

		float c = cos(i*3.1415926 / 180);

		for (float m = 0.0f;m < 0.001f;m += 0.0001f) {

			PointT point;

			point.x = center_x + (rad - m)*s;

			point.y = center_y + (rad - m)*c;

			point.z = center_z;

			cloud_pot->points.push_back(point);

		}
	}

	pcl::IterativeClosestPoint<PointT, PointT> icp;

	icp.setInputTarget(cloud_backup);

	icp.setInputSource(cloud_pot);

	icp.align(*cloud_icp);

	Eigen::Vector4f centroid;

	pcl::compute3DCentroid(*cloud_icp, centroid);

	//computeCentroid(*cloud_icp, pot_center);

	//cout << "pot_center in removePotOutlier: " << pot_center.x << ", " << pot_center.y << ", " << pot_center.z << endl;

	center_x = centroid[0];
	center_y = centroid[1];
	center_z = centroid[2];

	PointT search_p;

	cloud_out->clear();

	int r, g, b;

	float dis;

	for (size_t i = 0;i < cloud_backup->size();i++) {

		search_p = cloud_backup->points[i];

		r = (int)search_p.r;
		g = (int)search_p.g;
		b = (int)search_p.b;
		float exgreen = (2.f*g - r - b) / (r + g + b);

		dis = std::sqrt(pow(search_p.x - center_x, 2) + pow(search_p.y - center_y, 2));

		if (r == g && r == 255 && dis > rad) continue;

		if (exgreen > 0.2667 && search_p.z>center_z) {
			cloud_out->push_back(search_p);
			continue;
		}

		float dist_cone = rad - abs(center_z - search_p.z);

		float dis_2d = sqrt(pow((search_p.x - center_x), 2) + pow((search_p.y - center_y), 2));

		if(exp_NO != 6)
			if (dis_2d > dist_cone) continue;
			

		cloud_out->push_back(search_p);

	}

#ifdef visualize
	pcl::visualization::PCLVisualizer viewer("ICP demo");

	int v1(0);
	int v2(1);
	viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
	viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);

	viewer.registerPointPickingCallback(&pp_callback);

	viewer.addPointCloud(cloud_backup, to_string(cv::getTickCount()), v1);

	//pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_target_color_h(cloud_icp, 180, 20, 20);
	//viewer.addPointCloud(cloud_icp, cloud_target_color_h, "cloud_target_v1", v1);

	viewer.addPointCloud(cloud_out, to_string(cv::getTickCount()),v2);


	viewer.spin();
#endif
}

int regionGrowing(PointCloudT::Ptr cloud, PointCloudT::Ptr cloud_out) {
	cout << "regionGrowing\n";
	PointCloudT::Ptr cloud_green(new PointCloudT);
	PointCloudT::Ptr tmp(new PointCloudT);
	PointCloudT::Ptr soil(new PointCloudT);
	PointCloudT::Ptr cloud_backup(new PointCloudT);

	*cloud_backup += *cloud;
	cloud_out->clear();

	//remove outlier
	pcl::RadiusOutlierRemoval<PointT> outrem;
	outrem.setInputCloud(cloud_backup);
	outrem.setRadiusSearch(0.0005);
	outrem.setMinNeighborsInRadius(20);
	outrem.filter(*tmp);

	cout << "cloud in: " << cloud_backup->size() << endl;

	//downsample
	float leaf = 0.0004;

	pcl::VoxelGrid<PointT> sor;
	sor.setInputCloud(tmp);
	sor.setLeafSize(leaf, leaf, leaf);
	sor.filter(*tmp);

	cout << "cloud after preprocessing: " << tmp->size() << endl;

	int r, g, b;

	PointT p;
	for (int i = 0;i < tmp->size();i++) {
		p = tmp->points[i];

		if ((int)p.g == 0)
			soil->push_back(p);
		else
			cloud_green->push_back(p);
	}

	cout << "cloud_green: " << cloud_green->size() << endl;

	if (cloud_green->size() < 10)
		return -1;

	computeCentroid(*soil, p);
	float soil_z = p.z;
	cout << "soil_z: " << soil_z << endl;
	    
	PointT plant_center;
	computeCentroid(*cloud_green, plant_center);
	cout << "plant_center: " << plant_center.x << ", " << plant_center.y << ", " << plant_center.z << endl;

	if (cloud_green->size() < 100) {
		*cloud_out += *cloud_green;
		return 0;
	}

	//normal region growing get smooth parts
	pcl::search::Search<PointT>::Ptr tree = boost::shared_ptr<pcl::search::Search<PointT> >(new pcl::search::KdTree<PointT>);
	pcl::PointCloud <pcl::Normal>::Ptr normals(new pcl::PointCloud <pcl::Normal>);
	pcl::NormalEstimation<PointT, pcl::Normal> normal_estimator;
	normal_estimator.setSearchMethod(tree);
	normal_estimator.setInputCloud(cloud_green);
	normal_estimator.setKSearch(20);
	normal_estimator.compute(*normals);

	pcl::RegionGrowing<PointT, pcl::Normal> reg;
	reg.setMinClusterSize(0);
	reg.setMaxClusterSize(1000000);
	reg.setSearchMethod(tree);
	reg.setNumberOfNeighbours(20);
	reg.setInputCloud(cloud_green);
	//reg.setIndices (indices);
	reg.setInputNormals(normals);
	reg.setSmoothnessThreshold(30 / 180.0 * M_PI);
	reg.setCurvatureThreshold(0.01);

	std::vector <pcl::PointIndices> clusters;
	reg.extract(clusters);

	std::cout << "Region growing number of clusters " << clusters.size() << std::endl;

#if defined(visualize) && defined(leafsegmentation)

	pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud();

	//pcl::io::savePCDFile("paper_colored_cloud.pcd", *colored_cloud);

	leafSeg2(colored_cloud, clusters);

	return 0;
#endif

	int smooth_cluster_th;
	int ave_cluster_th = cloud_green->size() / 30; // in case the plant is so small  

	
	string phenotype = current_pot_id.substr(0, 3);
	cout << "current_pot_id: " << current_pot_id << " sub: " << phenotype << endl;

	if (phenotype == "bri") {
		ave_cluster_th = cloud_green->size() / 100;
	}

	smooth_cluster_th = std::min(ave_cluster_th, 300);
	cout << "smooth_cluster_th: " << smooth_cluster_th << endl;

	float  distance_to_center, distance_to_center_th, cluster_size;

	std::vector<Smooth> smooth_vec;
	std::vector<Smooth> smooth_vec_plant;

	for (int i = 0;i < clusters.size();i++) {
		tmp->clear();
		pcl::copyPointCloud(*cloud_green, clusters[i].indices, *tmp);
		pcl::computeCentroid(*tmp, p);

		Smooth smo;
		smo.cluster_indices = clusters[i].indices;
		smo.cluster_size = clusters[i].indices.size();
		smo.z_value = p.z;
		smooth_vec.push_back(smo);

	}

	std::sort(smooth_vec.begin(), smooth_vec.end(), scompare);

	PCA<PointT> pca;
	pca.setInputCloud(cloud_green);

	Eigen::Vector3f z_direction(0, 0, 1);
	Eigen::Vector3f major_vector;

	int k = 0;

	for (int i = 0;i < smooth_vec.size();i++) { //the first three biggest cluster

		Smooth s = smooth_vec.at(i);

		//cout << "smooth: " << i << ", size: " << s.cluster_size << " s.z: " << s.z_value << endl;

		if (s.z_value - plant_center.z > 0.01f) {
			smooth_vec.at(i).checked = true;

			continue;
		}

		if (s.cluster_size > 10){

			pca.setIndices(boost::make_shared<std::vector<int>>(s.cluster_indices));

			major_vector = pca.getEigenVectors().col(0);

			float abs_cos = abs(major_vector.dot(z_direction));

			if (acos(abs_cos) / PI*180.f < 20.f) continue;
		}

#ifdef visualize
		cout << "first three: \n";
		cout << "size: " << s.cluster_size << ", z: " << s.z_value << endl;
#endif

		for (auto &j : s.cluster_indices)
			cloud_out->push_back(cloud_green->points[j]);

		smooth_vec_plant.push_back(s);
		smooth_vec.at(i).checked = true;

		k++;

		if (k >= 3) break;

	} 
	

	if (cloud_out->size() < 300)
		return -1;
	else
	{

		pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud();
		leafSeg2(colored_cloud, clusters);  //include leafphenotype
		return -1;
	}

}

void registerRGBandPointCloud(cv::Mat & rgb, PointCloudT::Ptr scan_cloud, Eigen::Matrix4d & rgb_pose)
{
	cout << "registerRGBandPointCloud\n";
	PointCloudT::Ptr back_up(new PointCloudT);
	*back_up += *scan_cloud;

	std::vector<cv::Point3f> object_points(scan_cloud->points.size());
	for (int i = 0; i<scan_cloud->points.size(); i++)
	{
		object_points[i].x = scan_cloud->points[i].x;
		object_points[i].y = scan_cloud->points[i].y;
		object_points[i].z = scan_cloud->points[i].z;
	}

	Eigen::Matrix4d tmp_inverse = rgb_pose.inverse();

	cv::Mat rot, tvec, rvec;
	rot.create(3, 3, CV_32F); //rot.create(3, 3, CV_64F);
	tvec.create(3, 1, CV_32F);

	for (int y = 0; y < 3; y++)
		for (int x = 0; x < 3; x++)
			rot.at<float>(y, x) = tmp_inverse(y, x);

	tvec.at<float>(0, 0) = tmp_inverse(0, 3);
	tvec.at<float>(1, 0) = tmp_inverse(1, 3);
	tvec.at<float>(2, 0) = tmp_inverse(2, 3);

	cv::Rodrigues(rot, rvec);
	std::vector<cv::Point2f> img_points;

	//cout<<"after create\n";

	cv::projectPoints(object_points, rvec, tvec, kinect_rgb_camera_matrix_cv_, kinect_rgb_dist_coeffs_cv_, img_points);
	
	
	for (int i = 0; i < scan_cloud->points.size(); i++)
	{
		int x = std::round(img_points[i].x);
		int y = std::round(img_points[i].y);

		//	std::cout << x << "  " << y << "\n";

		if (x >= 0 && x < 1920 && y >= 0 && y < 1200)
		{
			scan_cloud->points[i].b = rgb.at<cv::Vec3b>(y, x).val[0];
			scan_cloud->points[i].g = rgb.at<cv::Vec3b>(y, x).val[1];
			scan_cloud->points[i].r = rgb.at<cv::Vec3b>(y, x).val[2];
		}
	}

	for (int i = 0; i < scan_cloud->points.size(); i++) {
		if (scan_cloud->points[i].r == 0)
			scan_cloud->points[i].r = 255;
	}

}


inline void downsample(PointCloudT::Ptr cloud, PointCloudT::Ptr cloud_out, float leaf_size) {

	cout << "before downsampling: " << cloud->size() << endl;

	PointCloudT::Ptr cloud_backup(new PointCloudT);
	*cloud_backup += *cloud;

	cloud_out->clear();

	pcl::VoxelGrid<PointT> sor;
	sor.setInputCloud(cloud_backup);
	sor.setLeafSize(leaf_size, leaf_size, leaf_size);
	sor.filter(*cloud_out);

#ifdef visualize
	cout << "leaf_size: " << leaf_size << endl;
	cout << "cloud after 0.0004 downsampling: " << cloud_out->size() << endl;
#endif

}


float getMeshArea(PointCloudT::Ptr cloud) {

	PointCloudT::Ptr cloud_backup(new PointCloudT);
	*cloud_backup += *cloud;
	cloud->clear();

	MovingLeastSquares<PointT, PointT> mls;
	mls.setInputCloud(cloud_backup);
	mls.setSearchRadius(0.001);
	mls.setPolynomialFit(true);
	mls.process(*cloud);

	downsample(cloud, cloud, 0.0004);

	pcl::NormalEstimation<PointT, pcl::Normal> n;
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
	tree->setInputCloud(cloud);
	n.setInputCloud(cloud);
	n.setSearchMethod(tree); //setKSearch(20);
	n.setRadiusSearch(0.001);
	n.compute(*normals);
	//* normals should not contain the point normals + surface curvatures

	// Concatenate the XYZ and normal fields*
	pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointXYZRGBNormal>);
	pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);

	pcl::GreedyProjectionTriangulation<pcl::PointXYZRGBNormal> gp3;
	pcl::PolygonMesh triangles;
	pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree2(new pcl::search::KdTree<pcl::PointXYZRGBNormal>);

	// Set the maximum distance between connected points (maximum edge length)
	gp3.setSearchRadius(0.002f); //0.001
	gp3.setMu(2.5);
	gp3.setMaximumNearestNeighbors(500);
	gp3.setMaximumSurfaceAngle(M_PI / 4); // 45 degrees
	gp3.setMinimumAngle(M_PI / 180); // 10 degrees
	gp3.setMaximumAngle(2 * M_PI / 3); // 120 degrees
	gp3.setNormalConsistency(false);

	// Get result
	gp3.setInputCloud(cloud_with_normals);
	gp3.setSearchMethod(tree2);
	gp3.reconstruct(triangles);


	double area = 0;
	size_t a, b, c;
	Eigen::Vector3d A, B, C, AB, AC, M;
	for (size_t i = 0; i < triangles.polygons.size(); ++i) {
		a = triangles.polygons[i].vertices[0];
		b = triangles.polygons[i].vertices[1];
		c = triangles.polygons[i].vertices[2];
		A(0) = cloud->points[a].x;
		A(1) = cloud->points[a].y;
		A(2) = cloud->points[a].z;
		B(0) = cloud->points[b].x;
		B(1) = cloud->points[b].y;
		B(2) = cloud->points[b].z;
		C(0) = cloud->points[c].x;
		C(1) = cloud->points[c].y;
		C(2) = cloud->points[c].z;
		AB = A - B;
		AC = A - C;
		M = AB.cross(AC);
		area += 0.5*(M.norm());
	}
	return area;

}


float getMicroCurvature(PointCloudT::Ptr cloud) {

	float radius=0.003f;

	//cin >> radius;

	//cout << "radius of curvature: " << radius << endl;

	PCA<PointT> pca;
	pca.setInputCloud(cloud);

	pcl::PointIndices::Ptr point_indices(new PointIndices);

	pcl::KdTreeFLANN<PointT> kdtree;

	kdtree.setInputCloud(cloud);

	std::vector<int> pointIdxRadiusSearch;
	std::vector<float> pointRadiusSquaredDistance;

	PointT searchPoint;
	
	float point_sd;
	float sum_sd = 0;
	for (int i = 0;i < cloud->size();i++) {
		point_indices->indices.clear();
		searchPoint = cloud->points[i];
		if (kdtree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0) {
			for (auto &j : pointIdxRadiusSearch)
				point_indices->indices.push_back(j);
		}

		if (point_indices->indices.size() < 3) continue;

		pca.setIndices(point_indices);

		Eigen::Vector3f eigen_values = pca.getEigenValues();
		point_sd = eigen_values(2) / (eigen_values(0) + eigen_values(1) + eigen_values(2));
		sum_sd += point_sd;

	}
	
	sum_sd = sum_sd / cloud->size();

	cout << "sum_sd: " << sum_sd << endl;

	return sum_sd;
}


inline int getEuclideanSize(PointCloudT::Ptr cloud) {

	pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);

	tree->setInputCloud(cloud);

	std::vector<pcl::PointIndices> cluster_indices;

	pcl::EuclideanClusterExtraction<PointT> ec;

	ec.setClusterTolerance(0.001); // two point 0.05mm  5e-5  0.0005

	ec.setMinClusterSize(0);

	ec.setMaxClusterSize(250000);

	ec.setSearchMethod(tree);

	ec.setInputCloud(cloud);

	ec.extract(cluster_indices);

	//	return cluster_indices.size();

	int j = 0;

	for (size_t i = 0;i < cluster_indices.size();i++) {

		if (cluster_indices[i].indices.size() > 5) {

			j++;

		}
	}

	return j;
}
void leafPhenotype(vector<PointCloudT::Ptr> &vec) {

	if (vec.size() < 1)
		return;

	cout << "get leaf phenotypes\n";
	
	for (int i = 0;i < vec.size();i++) {

		if (i == 3) break;

		PointCloudT::Ptr cloud(new PointCloudT);
		*cloud += *vec.at(i);

		cout << "leaf: " <<i<<": "<< cloud->size() << endl;

		Leaf lf;
		lf.num_size = cloud->size();
		lf.leaf_area = getMeshArea(cloud)* pow(100.f, 2);
		lf.leaf_smoothness = getMicroCurvature(cloud);

		PCA<PointT> pca;
		pca.setInputCloud(cloud);

		Eigen::Vector3f eigen_values = pca.getEigenValues();

		lf.leaf_curvature= eigen_values(2) / (eigen_values(0) + eigen_values(1) + eigen_values(2));

		PointCloud<PointXYZ>::Ptr temp(new PointCloud<PointXYZ>());

		pcl::MomentOfInertiaEstimation<PointXYZ> feature_extractor;
		pcl::PointXYZ min_point_OBB;
		pcl::PointXYZ max_point_OBB;
		pcl::PointXYZ position_OBB;
		Eigen::Matrix3f rotational_matrix_OBB;
		float major_value, middle_value, minor_value;
		Eigen::Vector3f major_vector, middle_vector, minor_vector;
		float voxel_size = 0.004f;
		ofstream outFile;

		pcl::copyPointCloud(*cloud, *temp);
		feature_extractor.setInputCloud(temp);
		feature_extractor.compute();

		feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);
		feature_extractor.getEigenValues(major_value, middle_value, minor_value);
		feature_extractor.getEigenVectors(major_vector, middle_vector, minor_vector);
		//cout << "minor_value: " << minor_value << ",middle_value: " << middle_value << ", major_value: " << major_value << endl;

		//lf.leaf_curvature = minor_value/ (minor_value + middle_value + major_value);

		Eigen::Vector3f major_vector_projected(major_vector(0), major_vector(1), 0);

		float abs_cos = abs(major_vector.dot(major_vector_projected) / major_vector.norm() / major_vector_projected.norm()); 

		lf.leaf_angle = acos(abs_cos) / PI*180.f;

		////////////////////////////////

		Eigen::Vector3f position(position_OBB.x, position_OBB.y, position_OBB.z);
		Eigen::Vector3f p1(min_point_OBB.x, min_point_OBB.y, min_point_OBB.z);
		Eigen::Vector3f p2(min_point_OBB.x, min_point_OBB.y, max_point_OBB.z);
		Eigen::Vector3f p4(max_point_OBB.x, min_point_OBB.y, min_point_OBB.z);
		Eigen::Vector3f p5(min_point_OBB.x, max_point_OBB.y, min_point_OBB.z);

		p1 = rotational_matrix_OBB * p1 + position;
		p2 = rotational_matrix_OBB * p2 + position;
		p4 = rotational_matrix_OBB * p4 + position;
		p5 = rotational_matrix_OBB * p5 + position;

		pcl::PointXYZ pt1(p1(0), p1(1), p1(2));
		pcl::PointXYZ pt2(p2(0), p2(1), p2(2));
		pcl::PointXYZ pt4(p4(0), p4(1), p4(2));
		pcl::PointXYZ pt5(p5(0), p5(1), p5(2));

		float dis12 = euclideanDistance(pt1, pt2);
		float dis14 = euclideanDistance(pt1, pt4);
		float dis15 = euclideanDistance(pt1, pt5);

		lf.leaf_length = dis14*100.f;
		lf.leaf_width = dis15*100.f;  //
									  // dis12*100.f;
		lf.leaf_boundingbox_volume = dis12*dis14*dis15*pow(100.f, 3);
		lf.aspect_ratio = lf.leaf_length / lf.leaf_width;

		ConvexHull<PointT> chull;
		PointCloudT::Ptr cloud_chull(new PointCloudT);
		chull.setInputCloud(cloud);
		chull.setComputeAreaVolume(true);
		chull.reconstruct(*cloud_chull);
		lf.leaf_convexhull_area = chull.getTotalArea() * pow(100.f, 2);
		lf.leaf_convexhull_volume = chull.getTotalVolume()*pow(100.f, 3);

		lf.leaf_solidity = lf.leaf_convexhull_volume / lf.leaf_boundingbox_volume;

		lf.leaf_convexity = lf.leaf_area / lf.leaf_convexhull_area;

#ifndef visualize
		outFile.open(leaf_measurement_path, ios::app); 

		if (leaf_file_first_line) {
			//title
			outFile.clear();
			outFile << "pot_id,sample_date,";
			for (int i = 1;i < 2;i++) { //i < 4
				//outFile << "leaf "<< i << ",";
				outFile << "num_size,leaf_angle,leaf_curvature,leaf_smoothness,";
				outFile << "leaf_length,leaf_width,aspect_ratio,";
				outFile << "leaf_area,leaf_convexhull_area,leaf_convexity,";
				outFile << "leaf_boundingbox_volume,leaf_convexhull_volume,leaf_solidity,";
			}	
			outFile << "\n";
			leaf_file_first_line = false;
		}	

		outFile << current_pot_id << "," << current_date << ",";
		outFile << lf.num_size << "," << lf.leaf_angle << "," << lf.leaf_curvature << "," << lf.leaf_smoothness << ",";
		outFile << lf.leaf_length << "," << lf.leaf_width << "," << lf.aspect_ratio << ",";
		outFile << lf.leaf_area << "," << lf.leaf_convexhull_area << "," << lf.leaf_convexity << ",";
		outFile << lf.leaf_boundingbox_volume << "," << lf.leaf_convexhull_volume << "," << lf.leaf_solidity;
		outFile << "\n";

		outFile.close();
#endif
	}
	
}

void leafSeg2(PointCloudT::Ptr cloud, vector <pcl::PointIndices> &clusters) {

	cout << "inside leafSeg2 clusters: " << clusters.size() << endl;

	PointCloudT::Ptr cloud_projected(new PointCloudT);

	// Create a set of planar coefficients with X=Y=0,Z=1
	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
	coefficients->values.resize(4);
	coefficients->values[0] = coefficients->values[1] = 0;
	coefficients->values[2] = 1.0;
	coefficients->values[3] = 0;

	// Create the filtering object
	pcl::ProjectInliers<PointT> proj;
	proj.setModelType(pcl::SACMODEL_PLANE);
	proj.setInputCloud(cloud);
	proj.setModelCoefficients(coefficients);
	proj.filter(*cloud_projected);


	PointT center;
	pcl::computeCentroid(*cloud_projected, center);
	PointCloudT::Ptr cloud_center(new PointCloudT);
	cloud_center->push_back(center);


	std::vector<Smooth> smooth_vec;

	PointCloudT::Ptr tmp(new PointCloudT);

	for (int i = 0; i < clusters.size(); i++) {

		Smooth smo;
		smo.cluster_indices = clusters[i].indices;
		smo.cluster_size = clusters[i].indices.size();
		
		smooth_vec.push_back(smo);
	}

	std::sort(smooth_vec.begin(), smooth_vec.end(), scompare);

	cout << "smooth_vec: " << smooth_vec.size() << endl;

	Eigen::Vector3f line_p, line_d, center_p, d;

	center_p = center.getArray3fMap();

	pcl::PCA<PointT> pca;

	pca.setInputCloud(cloud_projected);

	float dist;

	vector<PointCloudT::Ptr> leaf_vec;

	pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);

	tree->setInputCloud(cloud);

	pcl::EuclideanClusterExtraction<PointT> ec;

	ec.setInputCloud(cloud);

	string phenotype = current_pot_id.substr(0, 3);
	cout << "current_pot_id: " << current_pot_id << " sub: " << phenotype << endl;

	//if bri1 or bes1D or baf, do not use orientation
	if (phenotype == "bri" || phenotype == "bes") {
		for (int i = 0; i < smooth_vec.size(); i++) {

			std::vector<pcl::PointIndices> cluster_indices;

			ec.setClusterTolerance(0.001); // two point 0.05mm  5e-5  0.0005

			ec.setMinClusterSize(0);

			ec.setIndices(boost::make_shared<vector<int>>(smooth_vec[i].cluster_indices));

			ec.setMaxClusterSize(250000);

			ec.setSearchMethod(tree);

			ec.extract(cluster_indices);

			if (cluster_indices.size() < 1)
				continue;

			PointCloudT::Ptr leaf(new PointCloudT);
			pcl::copyPointCloud(*cloud, cluster_indices[0].indices, *leaf);
			leaf_vec.push_back(leaf);

			if (leaf_vec.size() >= 3)
				break;
		}		
	}
	else {
		for (int i = 0; i < smooth_vec.size(); i++) {

			cout<<"i: "<<i<<" cluster size: "<<smooth_vec[i].cluster_indices.size();
			//get the first cluster

			std::vector<pcl::PointIndices> cluster_indices;

			ec.setClusterTolerance(0.001); // two point 0.05mm  5e-5  0.0005

			ec.setMinClusterSize(0);

			ec.setIndices(boost::make_shared<vector<int>>(smooth_vec[i].cluster_indices));

			ec.setMaxClusterSize(250000);

			ec.setSearchMethod(tree);

			ec.extract(cluster_indices);

			cout<<"i: "<<i<<" cluster size: "<<smooth_vec[i].cluster_indices.size()<<" cluster_indices size: "<<cluster_indices.size()<<endl;

			if (cluster_indices.size() < 1)///////////////cluster_indices.size() < 3??? why??????
				continue;

			pcl::copyPointCloud(*cloud_projected, cluster_indices[0].indices, *tmp);

			cout<<"tmp->size(): "<<tmp->size()<<endl;

			if(tmp->size() < 10)
				continue;
			//get leaf candidates

			pcl::computeCentroid(*tmp, center);

			line_p = center.getArray3fMap();

			pca.setIndices(boost::make_shared<vector<int>>(smooth_vec[i].cluster_indices));

			line_d = pca.getEigenVectors().col(0); //major_vector

			d = line_d.cross(line_p - center_p);

			dist = d.norm() / line_d.norm();

			cout << "i " << i << " size: " << smooth_vec[i].cluster_size << " dist: " << dist << endl;

			if (dist < 0.005f) { //0.005f

				PointCloudT::Ptr leaf(new PointCloudT);

				pcl::copyPointCloud(*cloud, cluster_indices[0].indices, *leaf);  //smooth_vec[i].cluster_indices

				leaf_vec.push_back(leaf);

				cloud_center->push_back(center);

				if (leaf_vec.size() >= 3)
					break;
			}
		}
	}

	

	if (leaf_vec.size() < 1)
		return; //return leaf_vec;

#ifndef visualize
	//save pcd file
	PointXYZL pl;
	PointCloud<PointXYZL>::Ptr cloud_labeled(new PointCloud<PointXYZL>);
	int i = 0;
	for (auto &cloud : leaf_vec) {
		for (auto&p : cloud->points) {
			pl.x = p.x;
			pl.y = p.y;
			pl.z = p.z;
			pl.label = i;
			cloud_labeled->push_back(pl);
		}
		i++;
	}

	std::string CurrentOutputFolder = "";

	CurrentOutputFolder = current_exp_folder + "leaf_seg_eigen_dis2/" + current_pot_id;

	boost::filesystem::path dir(CurrentOutputFolder.c_str());

	if (boost::filesystem::create_directory(dir))
	{
		std::cerr << "Directory Created: " << CurrentOutputFolder << std::endl;
	}

	std::stringstream file_name;
	file_name << CurrentOutputFolder.c_str() << "/" << current_pot_id << "_" << current_date << "_leaf.pcd";

	pcl::io::savePCDFile(file_name.str(), *cloud_labeled);
#endif

	//return leaf_vec;


#ifdef visualize

	pcl::visualization::PCLVisualizer viewer0("ICP demo");

	int v1(0);
	int v2(1);
	viewer0.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
	viewer0.createViewPort(0.5, 0.0, 1.0, 1.0, v2);

	viewer0.registerPointPickingCallback(&pp_callback);

	viewer0.addPointCloud(cloud, to_string(cv::getTickCount()), v1);

#endif


//function leafphenotype

	if (leaf_vec.size() < 1)
		return;

	cout << "get leaf phenotypes\n";
	
	for (int i = 0;i < leaf_vec.size();i++) {

		if (i == 3) break;

		PointCloudT::Ptr cloud(new PointCloudT);
		*cloud += *leaf_vec.at(i);

		cout << "leaf: " <<i<<": "<< cloud->size() << endl;

		Leaf lf;
		lf.num_size = cloud->size();
		lf.leaf_area = getMeshArea(cloud)* pow(100.f, 2);
		lf.leaf_smoothness = getMicroCurvature(cloud);

		PCA<PointT> pca;
		pca.setInputCloud(cloud);

		Eigen::Vector3f eigen_values = pca.getEigenValues();

		lf.leaf_curvature= eigen_values(2) / (eigen_values(0) + eigen_values(1) + eigen_values(2));

		PointCloud<PointXYZ>::Ptr temp(new PointCloud<PointXYZ>());

		pcl::MomentOfInertiaEstimation<PointXYZ> feature_extractor;
		pcl::PointXYZ min_point_OBB;
		pcl::PointXYZ max_point_OBB;
		pcl::PointXYZ position_OBB;
		Eigen::Matrix3f rotational_matrix_OBB;
		float major_value, middle_value, minor_value;
		Eigen::Vector3f major_vector, middle_vector, minor_vector;
		float voxel_size = 0.004f;
		ofstream outFile;

		pcl::copyPointCloud(*cloud, *temp);
		feature_extractor.setInputCloud(temp);
		feature_extractor.compute();

		feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);
		feature_extractor.getEigenValues(major_value, middle_value, minor_value);
		feature_extractor.getEigenVectors(major_vector, middle_vector, minor_vector);
		//cout << "minor_value: " << minor_value << ",middle_value: " << middle_value << ", major_value: " << major_value << endl;

		//lf.leaf_curvature = minor_value/ (minor_value + middle_value + major_value);

		Eigen::Vector3f major_vector_projected(major_vector(0), major_vector(1), 0);

		float abs_cos = abs(major_vector.dot(major_vector_projected) / major_vector.norm() / major_vector_projected.norm()); 

		lf.leaf_angle = acos(abs_cos) / PI*180.f;

		////////////////////////////////

		Eigen::Vector3f position(position_OBB.x, position_OBB.y, position_OBB.z);
		Eigen::Vector3f p1(min_point_OBB.x, min_point_OBB.y, min_point_OBB.z);
		Eigen::Vector3f p2(min_point_OBB.x, min_point_OBB.y, max_point_OBB.z);
		Eigen::Vector3f p4(max_point_OBB.x, min_point_OBB.y, min_point_OBB.z);
		Eigen::Vector3f p5(min_point_OBB.x, max_point_OBB.y, min_point_OBB.z);

		p1 = rotational_matrix_OBB * p1 + position;
		p2 = rotational_matrix_OBB * p2 + position;
		p4 = rotational_matrix_OBB * p4 + position;
		p5 = rotational_matrix_OBB * p5 + position;
		
		/*
		Eigen::Vector3f p1 (min_point_OBB.x, min_point_OBB.y, min_point_OBB.z);
		Eigen::Vector3f p2 (min_point_OBB.x, min_point_OBB.y, max_point_OBB.z);
		Eigen::Vector3f p3 (max_point_OBB.x, min_point_OBB.y, max_point_OBB.z);
		Eigen::Vector3f p4 (max_point_OBB.x, min_point_OBB.y, min_point_OBB.z);
		Eigen::Vector3f p5 (min_point_OBB.x, max_point_OBB.y, min_point_OBB.z);
		Eigen::Vector3f p6 (min_point_OBB.x, max_point_OBB.y, max_point_OBB.z);
		Eigen::Vector3f p7 (max_point_OBB.x, max_point_OBB.y, max_point_OBB.z);
		Eigen::Vector3f p8 (max_point_OBB.x, max_point_OBB.y, min_point_OBB.z);

		p1 = rotational_matrix_OBB * p1 + position;
		p2 = rotational_matrix_OBB * p2 + position;
		p3 = rotational_matrix_OBB * p3 + position;
		p4 = rotational_matrix_OBB * p4 + position;
		p5 = rotational_matrix_OBB * p5 + position;
		p6 = rotational_matrix_OBB * p6 + position;
		p7 = rotational_matrix_OBB * p7 + position;
		p8 = rotational_matrix_OBB * p8 + position;*/


		pcl::PointXYZ pt1(p1(0), p1(1), p1(2));
		pcl::PointXYZ pt2(p2(0), p2(1), p2(2));
		pcl::PointXYZ pt4(p4(0), p4(1), p4(2));
		pcl::PointXYZ pt5(p5(0), p5(1), p5(2));

		float dis12 = euclideanDistance(pt1, pt2);
		float dis14 = euclideanDistance(pt1, pt4);
		float dis15 = euclideanDistance(pt1, pt5);

		lf.leaf_length = dis14*100.f;
		lf.leaf_width = dis15*100.f;  //
									  // dis12*100.f;
		lf.leaf_boundingbox_volume = dis12*dis14*dis15*pow(100.f, 3);
		lf.aspect_ratio = lf.leaf_length / lf.leaf_width;

		ConvexHull<PointT> chull;
		PointCloudT::Ptr cloud_chull(new PointCloudT);
		chull.setInputCloud(cloud);
		chull.setComputeAreaVolume(true);
		chull.reconstruct(*cloud_chull);
		lf.leaf_convexhull_area = chull.getTotalArea() * pow(100.f, 2);
		lf.leaf_convexhull_volume = chull.getTotalVolume()*pow(100.f, 3);

		lf.leaf_solidity = lf.leaf_convexhull_volume / lf.leaf_boundingbox_volume;

		lf.leaf_convexity = lf.leaf_area / lf.leaf_convexhull_area;

#ifndef visualize
		outFile.open(leaf_measurement_path, ios::app); 

		if (leaf_file_first_line) {
			//title
			outFile.clear();
			outFile << "pot_id,sample_date,";
			for (int i = 1;i < 2;i++) { //i < 4
				//outFile << "leaf "<< i << ",";
				outFile << "num_size,leaf_angle,leaf_curvature,leaf_smoothness,";
				outFile << "leaf_length,leaf_width,aspect_ratio,";
				outFile << "leaf_area,leaf_convexhull_area,leaf_convexity,";
				outFile << "leaf_boundingbox_volume,leaf_convexhull_volume,leaf_solidity,";
			}	
			outFile << "\n";
			leaf_file_first_line = false;
		}	

		outFile << current_pot_id << "," << current_date << ",";
		outFile << lf.num_size << "," << lf.leaf_angle << "," << lf.leaf_curvature << "," << lf.leaf_smoothness << ",";
		outFile << lf.leaf_length << "," << lf.leaf_width << "," << lf.aspect_ratio << ",";
		outFile << lf.leaf_area << "," << lf.leaf_convexhull_area << "," << lf.leaf_convexity << ",";
		outFile << lf.leaf_boundingbox_volume << "," << lf.leaf_convexhull_volume << "," << lf.leaf_solidity;
		outFile << "\n";

		outFile.close();
#else
		cout <<"length: " << lf.leaf_length << " width: " << lf.leaf_width << endl;
		viewer0.addPointCloud(cloud, to_string(cv::getTickCount()), v2);

		pcl::computeCentroid(*cloud, center);
		string tmpstr = "l: " + to_string(lf.leaf_length).substr(0,4) + ", w: " + to_string(lf.leaf_width).substr(0,4);

		viewer0.addText3D(tmpstr, center, 0.003, 1.0, 1.0, 1.0, to_string(cv::getTickCount()), v1);
		/*
		viewer0.addLine (pt1, pt2, 1.0, 0.0, 0.0, to_string(cv::getTickCount()));
		viewer0.addLine (pt1, pt4, 0.0, 1.0, 0.0, to_string(cv::getTickCount()));
		viewer0.addLine (pt1, pt5, 0.0, 0.0, 1.0, to_string(cv::getTickCount()));*/
		
		Eigen::Vector4f min_pt, max_pt;
		float line_width = 2;
		pcl::getMinMax3D(*cloud, min_pt, max_pt);
		string name = to_string(i);
		viewer0.addCube (min_pt(0), max_pt(0), min_pt(1),max_pt(1), min_pt(2), max_pt(2), 1.0, 1.0, 0.0, name, v2);
		viewer0.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME,name); 
		viewer0.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, line_width,name); 


#endif
	}



#ifdef visualize
	viewer0.spin();
#endif

}

