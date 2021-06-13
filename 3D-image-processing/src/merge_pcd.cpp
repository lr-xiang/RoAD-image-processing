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
#include <pcl/console/time.h>   

#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/radius_outlier_removal.h>

#include <pcl/segmentation/extract_clusters.h> 
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
#define merge 
#define PI 3.14159265

int exp_NO = 15;
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
	
	float occupancy = 1.f;

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
	PointCloudT::Ptr cloud_out, int scan_id);

void
print4x4Matrix(const Eigen::Matrix4d & matrix);

int regionGrowing(PointCloudT::Ptr cloud, PointCloudT::Ptr cloud_out);

int registerRGBandPointCloud(cv::Mat & rgb, PointCloudT::Ptr scan_cloud, Eigen::Matrix4d & rgb_pose);

inline void downsample(PointCloudT::Ptr cloud, PointCloudT::Ptr cloud_out, float leaf_size);

inline void upsample(PointCloudT::Ptr cloud, PointCloudT::Ptr cloud_out);

vector<std::string> traits_3d_vec{};
vector<std::string> traist_2d_vec{};


string current_exp_folder;


int scan_num = 4; //used in load data

string water_type = "";
string phenotype = "";

//usage: ./merge /media/lietang/easystore1/RoAD/exp15/ 15
int
main(int argc,
	char* argv[])
{

	current_exp_folder = argv[1];
	
	exp_NO = stoi(argv[2]);


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
	if(exp_NO ==5)
		rgb_pose_file = "rgb_pose_exp5.yml";	
	if(exp_NO ==6)
		rgb_pose_file = "rgb_pose_exp6.yml";	
	if(exp_NO ==10)
		rgb_pose_file = "rgb_pose_exp10.yml";
	if(exp_NO ==13)
		rgb_pose_file = "rgb_pose_exp13.yml";
	if(exp_NO >= 15)
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
	
		current_pot_id = pot_labels[i];
		cout<<"i: "<<i<<endl;
		//get waterType "W" or "H"
		std::vector<std::string> str_vec;
		boost::split(str_vec, current_pot_id, boost::is_any_of("_"));

		water_type = str_vec[1];
		cout<<"water_type: "<<water_type<<endl;

		
		phenotype = current_pot_id.substr(0, 3);
		cout << "current_pot_id: " << current_pot_id << " sub: " << phenotype << endl;
				
		for (int j = 0;j < date.size();j++) {
	
			current_date = date[j];
			cout<<"i: "<<i<<" j: "<<j<<endl;		

			std::vector<PointCloudT::Ptr, Eigen::aligned_allocator<PointCloudT::Ptr> > data;

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
	if (image.empty())                     
	{
		cout << "Could not open or find the image" << std::endl;
		return -1;
	}
	cout<<cloud->size()<<endl;
	rgb_pose = rgb_pose_0;
	if(registerRGBandPointCloud(image, cloud, rgb_pose)<0)
		return -1;
	data.push_back(cloud);

	cout << "data 0 : " << data.at(0)->size() << endl;


	for (int i = scan_num;i >1;i--) {
	
	//without scan 1, incomplete

		temp_path = path + "_scan_" + to_string(i) + ".pcd";

		PointCloudT::Ptr cloud_source(new PointCloudT);
		
		if(i==1)
			rgb_pose = rgb_pose_1;
		else if(i==2)
			rgb_pose = rgb_pose_2;
		else if(i==3)
			rgb_pose = rgb_pose_3;
		else if(i==4)
			rgb_pose = rgb_pose_4;

		if (pcl::io::loadPCDFile<PointT>(temp_path, *cloud_source) == -1) //* load the file
		{

			PCL_ERROR("Couldn't read file \n");
			std::cout << temp_path << std::endl;
			return -1;

		}
		

		registerRGBandPointCloud(image, cloud_source, rgb_pose);
		data.push_back(cloud_source);

		cout << "data " << scan_num + 1 - i << " : " << data.at(scan_num + 1 - i)->size() << endl;

	}
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

	removePotOutlier(cloud_target, cloud_target, 0);

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

			removePotOutlier(cloud_source, cloud_source, scan_num + 1 - i);

			cout << "source cloud: " << cloud_source->size() << " points." << endl;

			time.tic();

			downsample(cloud_source, cloud_source_downsample, 0.0004);

			icp.setInputTarget(cloud);

			icp.setInputSource(cloud_source_downsample);

			icp.align(*cloud_icp);

			*cloud += *cloud_icp;


			cout << "round " << i << " finished registration in " << time.toc() << " ms.\n" << endl;;

			transformation_matrix = icp.getFinalTransformation().cast<double>();


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
	PointCloudT::Ptr cloud_out, int scan_id) {
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

	float dis, dis_2d;

	for (size_t i = 0;i < cloud_backup->size();i++) {

		search_p = cloud_backup->points[i];

		r = (int)search_p.r;
		g = (int)search_p.g;
		b = (int)search_p.b;
		float exgreen = (2.f*g - r - b) / (r + g + b);

		dis = std::sqrt(pow(search_p.x - center_x, 2) + pow(search_p.y - center_y, 2) + pow(search_p.z - center_z, 2));

		dis_2d = sqrt(pow((search_p.x - center_x), 2) + pow((search_p.y - center_y), 2));


		if (r == g && r == 255 && dis_2d > rad) continue;

		if (exgreen > 0.2667 && search_p.z>center_z) {
			cloud_out->push_back(search_p);
			continue;
		}

		float dist_cone = rad - abs(center_z - search_p.z);
			
		if (scan_id == 3 || scan_id == 2 )
			if(dis_2d > dist_cone) continue; 

		cloud_out->push_back(search_p);

	}
//

#ifdef visualize
	pcl::visualization::PCLVisualizer viewer("ICP demo");

	int v1(0);
	int v2(1);
	viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
	viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);

	viewer.registerPointPickingCallback(&pp_callback);

	viewer.addPointCloud(cloud_backup, to_string(cv::getTickCount()), v1);

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
	int r_th = 200;
	int k_th = 3;

	for (int i = 0;i < tmp->size();i++) {
		p = tmp->points[i];
			
		if((int)p.g > 0 && (int)p.r < r_th)
			cloud_green->push_back(p);
		else
			soil->push_back(p);
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
	std::vector <pcl::PointIndices> clusters;
		

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


	reg.extract(clusters);

	std::cout << "Region growing number of clusters " << clusters.size() << std::endl;

	pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = reg.getColoredCloud();
	
	int smooth_cluster_th;
	int ave_cluster_th = cloud_green->size() / 30; // in case the plant is so small  

	cout << "current_pot_id: " << current_pot_id << " sub: " << phenotype << endl;

	if (phenotype == "bri"  ) {
		ave_cluster_th = cloud_green->size() / 100;
	}

	smooth_cluster_th = std::min(ave_cluster_th, 300);
	cout << "smooth_cluster_th: " << smooth_cluster_th << endl;

	float  distance_to_center, distance_to_center_th, cluster_size;

	std::vector<Smooth> smooth_vec;
	std::vector<Smooth> smooth_vec_plant;

	//sort out by size 
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
	
	float height_th = 0.01f;

	for (int i = 0;i < smooth_vec.size();i++) { 

		Smooth s = smooth_vec.at(i);

		if (s.z_value - plant_center.z > height_th) {
			smooth_vec.at(i).checked = true;

			continue;
		}

		if (s.cluster_size > 10){

			pca.setIndices(boost::make_shared<std::vector<int>>(s.cluster_indices));

			major_vector = pca.getEigenVectors().col(0);

			float abs_cos = abs(major_vector.dot(z_direction));

			if (acos(abs_cos) / PI*180.f < 20.f) continue;
		}


		for (auto &j : s.cluster_indices)
			cloud_out->push_back(cloud_green->points[j]);

		smooth_vec_plant.push_back(s);
		smooth_vec.at(i).checked = true;

		k++;

		if (k >= k_th) break;

	} 
	cout<<"cloud_out size: "<<cloud_out->size()<<endl;
	
	Eigen::Vector4f min_pt, max_pt;
	Eigen::Vector4f cluster_min_pt, cluster_max_pt;

	pcl::getMinMax3D(*cloud_out, min_pt, max_pt);

	float plant_radius = (max_pt(0) + max_pt(1) - min_pt(0) - min_pt(1)) / 4.f;

	cout << "cloud green\n";
	cout << "plant_center: " << plant_center.x << ", " << plant_center.y << ", " << plant_center.z << endl;
	cout << "plant_radius: " << plant_radius << endl;
	
	for (int i = 0;i < smooth_vec.size();i++) {

		distance_to_center = 0;

		Smooth s = smooth_vec.at(i);

		if (s.checked)continue;

		for (auto &j : s.cluster_indices)
			distance_to_center += euclideanDistance(cloud_green->points[j], plant_center);

		s.distance_to_center = distance_to_center / s.cluster_size;;

		float scaled_cluster_th = s.distance_to_center / plant_radius*smooth_cluster_th;

		s.scaled_cluster_th = std::max((int)scaled_cluster_th, 50); 
			
		if ((s.cluster_size >300 || s.cluster_size > s.scaled_cluster_th) 
		&& (s.cluster_size > smooth_vec.at(0).cluster_size*0.2f) //remove curve pcd around soil || reseve long leaf
		) { 
			smooth_vec_plant.push_back(s);
			smooth_vec.at(i).checked = true;
			for (auto &j : s.cluster_indices)
				cloud_out->push_back(cloud_green->points[j]);
		}
		else {
			for (auto &j : s.cluster_indices)
				soil->push_back(cloud_green->points[j]);
		}

	}

	cout << "cloud_out: " << cloud_out->size() << endl;

#ifdef visualize
	pcl::visualization::PCLVisualizer viewer("ICP demo");

	int v1(0);
	int v2(1);
	viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
	viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);

	viewer.registerPointPickingCallback(&pp_callback);

	viewer.addPointCloud(colored_cloud, to_string(cv::getTickCount()), v1); // soil
#endif

	cloud_backup->clear();
	*cloud_backup += *cloud_out;
	cloud_out->clear();
	for (int i = 0;i < cloud_backup->size();i++) {
		p = cloud_backup->points[i];
		r = (int)p.r;
		g = (int)p.g;
		b = (int)p.b;
		if (r > 250 && b == 0)
			continue;
		cloud_out->push_back(p);
	}

	cout << "cloud_out: " << cloud_out->size() << endl;

#ifdef visualize
	viewer.addPointCloud(cloud_out, to_string(cv::getTickCount()), v2);

	viewer.spin();
#endif
	return 1;
}

int registerRGBandPointCloud(cv::Mat & rgb, PointCloudT::Ptr scan_cloud, Eigen::Matrix4d & rgb_pose)
{

	cout << "registerRGBandPointCloud\n";
	cout<<"rgb_pose: "<<rgb_pose<<endl;
	
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

//	cout<<"before create\n";


	int count = 0;
	
	for (int i = 0; i < scan_cloud->points.size(); i++) {
		if (scan_cloud->points[i].r == 0)
			scan_cloud->points[i].r = 255;
		else
			count++;
	}

	if (count<10)
		return -1;
		
	return 1;

}


inline void downsample(PointCloudT::Ptr cloud, PointCloudT::Ptr cloud_out, float leaf_size) {

	cout << "before downsampling: " << cloud->size() << endl;

	PointCloudT::Ptr cloud_backup(new PointCloudT);
	*cloud_backup += *cloud;

	cloud_out->clear();

	//downsample


	pcl::VoxelGrid<PointT> sor;
	sor.setInputCloud(cloud_backup);
	sor.setLeafSize(leaf_size, leaf_size, leaf_size);
	sor.filter(*cloud_out);

#ifdef visualize
	cout << "leaf_size: " << leaf_size << endl;
	cout << "cloud after 0.0004 downsampling: " << cloud_out->size() << endl;
#endif

}

inline void upsample(PointCloudT::Ptr cloud, PointCloudT::Ptr cloud_out) {

	cout << "before upsampling: " << cloud->size() << endl;

	PointCloudT::Ptr cloud_backup(new PointCloudT);
	*cloud_backup += *cloud;

	cloud_out->clear();

	//upsample
	pcl::MovingLeastSquares<PointT, PointT> mls;
	mls.setInputCloud(cloud_backup);
	mls.setSearchRadius(0.001);
	mls.setPolynomialFit(true);
	mls.setPolynomialOrder(2);
	mls.setUpsamplingMethod(MovingLeastSquares<PointT, PointT>::SAMPLE_LOCAL_PLANE);
	mls.setUpsamplingRadius(0.0005); //0.0005
	mls.setUpsamplingStepSize(0.0004); //0.0004
	mls.process(*cloud_out);
	cout << "cloud after upsampling: " << cloud_out->size() << endl;

	pcl::visualization::PCLVisualizer viewer("demo");

	int v1(0);
	int v2(1);
	viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
	viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);

	viewer.registerPointPickingCallback(&pp_callback);

	viewer.addPointCloud(cloud_backup, to_string(cv::getTickCount()), v1);
	viewer.addPointCloud(cloud_out, to_string(cv::getTickCount()), v2);

	viewer.spin();

}

