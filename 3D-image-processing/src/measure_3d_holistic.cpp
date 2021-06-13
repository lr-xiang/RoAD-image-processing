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

#define measure  
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

};

struct Rosette {

	int num_size = -1;

	float plant_height = -1;
	
	float length_of_boundingbox = -1;

	float width_of_boundingbox = -1;

	float height_of_boundingbox = -1;

	float holistic_aspect_ratio = -1; // width/length

	float area = -1;

	float convex_area = -1;

	float holistic_area_convexity = -1; // area/convex_area
	
	float volume = -1;
	
	float convex_hull_volume = -1;

	float boundingbox_volume = -1;
	
	float holistic_solidity = -1;

	float holistic_curvature = -1;

	float micro_curvature = -1;
	
	float area_based_volume = -1.f;

	float area_based_solidity = -1.f;
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

struct Arab {
	int exp_no;
	string pot_id;
	int pot_num;
	string geno;
	string treatment;
	float target_weight;
	float moisture_level;
	float replication;
	string sample_date;
//3d
	float num_size = -1.f; //
	float plant_height = -1.f;
	float holistic_aspect_ratio = -1.f;
	float length_of_boundingbox = -1.f; //
	float width_of_boundingbox = -1.f; //
	float height_of_boundingbox = -1.f; //
	float area = -1.f;
	float convex_area = -1.f; //
	float holistic_area_convexity = -1.f;
	float convex_hull_volume = -1.f;
	float boundingbox_volume = -1.f;//
	float holistic_solidity = -1.f;
	float holistic_curvature = -1.f;
	float micro_curvature = -1.f;
	float area_based_volume = -1.f;
	
	int abs_day;
	float abs_area = 0;
	float abs_height = 0;
	float abs_volume = 0;
	float abs_num_size = 0; 
	float abs_length_of_boundingbox = 0; //
	float abs_width_of_boundingbox = 0; //
	float abs_height_of_boundingbox = 0; //
	float abs_convex_area = 0; //
	float abs_boundingbox_volume = 0;//


	float fresh_weight = -1.f;
	float dry_weight = -1.f;
	//float solidity2d = -1.f;


//2d traits
	float area_2d  = -1.f;
	float convex_area_2d = -1.f;
	float solidity = -1.f;
	float perimeter = -1.f;
	float boundingbox_area_2d = -1.f;
	float aspect_ratio = -1.f;
	float rectangularity = -1.f;
	float circularity = -1.f;
	float averageR = -1.f;
	float averageG = -1.f;
	float averageB = -1.f;
	float hsvH = -1.f;
	float hsvS = -1.f;
	float hsvV = -1.f;
	float labL = -1.f;
	float labA = -1.f;
	float labB = -1.f;

	float abs_area_2d = -1.f;
	float abs_convex_area_2d = -1.f;
	float abs_perimeter = -1.f;
	
//indivdual leaf traits
	float leaf_num_size = -1.f;
	float leaf_angle = -1.f;
	float leaf_curvature = -1.f;
	float leaf_smoothness = -1.f;
	float leaf_length = -1.f;
	float leaf_width = -1.f;
	float leaf_aspect_ratio = -1.f;
	float leaf_area = -1.f;
	float leaf_convexhull_area = -1.f;
	float leaf_convexity = -1.f;
	float leaf_boundingbox_volume = -1.f;
	float leaf_convexhull_volume = -1.f;
	float leaf_solidity = -1.f;

};

bool scompare(const Smooth &l1, const Smooth &l2) { return l1.cluster_size > l2.cluster_size; }
bool zcompare(const Smooth &l1, const Smooth &l2) { return l1.z_value < l2.z_value; }
bool cloud_size_compare(const PointCloudT::Ptr &p1, const PointCloudT::Ptr &p2) { return p1->size()>p2->size(); }

float pre_x = 0, pre_y = 0, pre_z = 0, Dist = 0;
void pp_callback(const pcl::visualization::PointPickingEvent& event);

void
print4x4Matrix(const Eigen::Matrix4d & matrix);

int regionGrowing(PointCloudT::Ptr cloud, PointCloudT::Ptr cloud_out);

inline void downsample(PointCloudT::Ptr cloud, PointCloudT::Ptr cloud_out, float leaf_size);

float getMeshArea(PointCloudT::Ptr cloud);

float getVoxelVolume(PointCloudT::Ptr cloud);
float getSlicedVolume(PointCloudT::Ptr cloud);

float getPercentileHeight(PointCloudT::Ptr cloud);

float getMicroCurvature(PointCloudT::Ptr cloud);

vector<PointCloudT::Ptr> leafSeg(PointCloudT::Ptr cloud);

inline int getEuclideanSize(PointCloudT::Ptr cloud);

void holisticMeasurement(PointCloudT::Ptr cloud);

vector<std::string> traits_vec{"exp_no,geno,treatment,\
target_weight,moisture_level,replication,pot_num,sample_date,\
num_size,plant_height,holistic_aspect_ratio,\
length_of_boundingbox,width_of_boundingbox,height_of_boundingbox,\
area,convex_area,holistic_area_convexity,\
convex_hull_volume,boundingbox_volume,\
holistic_solidity,holistic_curvature,micro_curvature,\
abs_day,abs_area,abs_height,\
abs_volume,abs_num_size,abs_length_of_boundingbox,\
abs_width_of_boundingbox,abs_height_of_boundingbox,abs_convex_area,abs_boundingbox_volume,\
area_2d,convex_area_2d,solidity,perimeter,boundingbox_area_2d,\
aspect_ratio,rectangularity,circularity,averageR,averageG,averageB,hsvH,hsvS,hsvV,labL,labA,labB,\
abs_area_2d,abs_convex_area_2d,\
leaf_num_size,leaf_angle,leaf_curvature,leaf_smoothness,\
leaf_length,leaf_width,leaf_aspect_ratio,leaf_area,\
leaf_convexhull_area,leaf_convexity,leaf_boundingbox_volume,leaf_convexhull_volume,leaf_solidity\
"};



vector<std::string> traits_3d_vec{};
vector<std::string> traist_2d_vec{};


string current_exp_folder, holistic_measurement_path,leaf_measurement_path, traits_2d_path;

float leave_thickness = 0.002f; //used in area_based_volume and sliced volume

//./measure /media/lietang/easystore1/RoAD/exp15/ 15

int
main(int argc,
	char* argv[])
{

	current_exp_folder = argv[1];
	
	exp_NO = stoi(argv[2]);

	holistic_measurement_path = current_exp_folder + "3d_images/traits_holistic_3d.csv";

	leaf_measurement_path = current_exp_folder + "leaf_seg_eigen_dis2/individual_leaf_traits.csv";

	traits_2d_path = current_exp_folder + "2d_traits.csv";

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
	
				
	cv::FileStorage fs_pose("rgb_pose.yml", cv::FileStorage::READ);
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


	//for (int i = 0;i < pot_labels.size();i++) {

	for (int i = 0;i < pot_labels.size();i++) { // to improve efficiency, change this part
		for (int j = 0;j < date.size();j++) {

			current_pot_id = pot_labels[i];
			current_date = date[j];
			cout<<"i: "<<i<<" j: "<<j<<endl;
			
#ifdef measure
			cout << "measuring " << current_pot_id << " on " << current_date << endl;

			PointCloudT::Ptr cloud(new PointCloudT);

			std::stringstream path;

			path << current_exp_folder<<"3d_images/";

			path << current_pot_id << "/";
			path << current_pot_id << "_" << current_date<<".pcd";

			if (pcl::io::loadPCDFile<PointT>(path.str(), *cloud) == -1) //* load the file
			{

				PCL_ERROR("Couldn't read file test_pcd.pcd \n");
				continue;
			}

			holisticMeasurement(cloud);
			continue;
			
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



inline void downsample(PointCloudT::Ptr cloud, PointCloudT::Ptr cloud_out, float leaf_size) {

	//cout << "before downsampling: " << cloud->size() << endl;

	PointCloudT::Ptr cloud_backup(new PointCloudT);
	*cloud_backup += *cloud;

	cloud_out->clear();

	pcl::VoxelGrid<PointT> sor;
	sor.setInputCloud(cloud_backup);
	sor.setLeafSize(leaf_size, leaf_size, leaf_size);
	sor.filter(*cloud_out);

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

	// Set typical values for the parameters
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

float getVoxelVolume(PointCloudT::Ptr cloud){

	cout<<"get cloud voxel volume\n";
	cout<<"cloud size: "<<cloud->size()<<endl;
	
	float delta_d = 0.0006f;
	
	Eigen::Vector4f min_pt, max_pt;
	pcl::getMinMax3D(*cloud, min_pt, max_pt);
	
	float delta_x = max_pt[0] - min_pt[0]; 		
	float delta_y = max_pt[1] - min_pt[1]; 
	float delta_z = max_pt[2] - min_pt[2]; 
	
	cout<<"get xyz count\n";
	float x_count = delta_x/delta_d;
	float y_count = delta_y/delta_d;
	float z_count = delta_z/delta_d;
	cout << "x_count: "<< x_count << " y_count: "<< y_count << " z_count: "<< z_count << endl;
		
	int x_len = int(x_count+1);
	int y_len = int(y_count+1);
	int z_len = int(z_count+1);
	cout << "x_len: "<< x_len << " y_len: "<< y_len << " z_len: "<< z_len << endl;
	
	cout<<"create a map\n"; 
	int map[x_len][y_len][z_len];
	//static int map[100][100][100];
	
	float count = 0.f;

    cout<<"initialize the 3d array\n";
    	
	for (int i = 0; i < x_len; ++i) 
    { 
        for (int j = 0; j < y_len; ++j) 
        { 
            for (int k = 0; k < z_len; ++k) 
                map[i][j][k] = 0;
        } 
    } 

/*
	for (int i = 0; i < x_len; ++i) 
    { 
        for (int j = 0; j < y_len; ++j) 
        { 
            for (int k = 0; k < z_len; ++k) 
            { 
                cout << "Element at x[" << i << "][" << j 
                     << "][" << k << "] = " << map[i][j][k] 
                     << endl; 
            } 
        } 
    } 
    cout<<"count: "<<count<<endl;
    */

    cout<<"count in pcd\n";
    PointT p;
    float a, b, c;
    for (int i=0; i<cloud->size();i++){
    	p = cloud->points[i];
    	a = (p.x - min_pt[0])/delta_d;
    	b = (p.y - min_pt[1])/delta_d;
    	c = (p.z - min_pt[2])/delta_d;
    	map[int(a)][int(b)][int(c)] = 1;
    }
/*
    for (int i = 0; i < x_len; ++i) 
    { 
        for (int j = 0; j < y_len; ++j) 
        { 
            for (int k = 0; k < z_len; ++k) 
            { 
                cout << "Element at x[" << i << "][" << j 
                     << "][" << k << "] = " << map[i][j][k] 
                     << endl; 
            } 
        } 
    } 
 */   	
    count = 0;
    for (int i = 0; i < x_len; ++i) 
    { 
        for (int j = 0; j < y_len; ++j) 
        { 
            for (int k = 0; k < z_len; ++k) 
            { 
                if( map[i][j][k] == 1)
                	count++;
            } 
        } 
    } 
    
	float voxel_volume = count*pow(delta_d*100.f,3);
	float total_volume = x_count*y_count*z_count*pow(delta_d*100.f,3);
	cout << "delta_x: "<< delta_x << " delta_y: "<< delta_y << " delta_z: "<< delta_z << endl;

	cout<<"count: "<<count<<endl;
	cout <<"voxel_volume: "<<voxel_volume<<" total_volume: "<<total_volume<<endl;
	
	return voxel_volume;

}

float getSlicedVolume(PointCloudT::Ptr cloud){

	cout<<"get cloud sliced volume\n";
	cout<<"cloud size: "<<cloud->size()<<endl;
	
	float slices_N; // = 5.f;
	
	Eigen::Vector4f min_pt, max_pt;
	pcl::getMinMax3D(*cloud, min_pt, max_pt);
	
	float slice_height = leave_thickness; //0.005f
	slices_N = (max_pt[2] - min_pt[2])/slice_height;
	cout<<"slices_N: "<<slices_N<< " slice_height: "<<slice_height<<endl;
	
	PointCloudT::Ptr cloud_filtered(new PointCloudT);
	PointCloudT::Ptr cloud_projected(new PointCloudT);
	PointCloudT::Ptr colored_cloud(new PointCloudT);
	
	float slice_volume = 0.f; //in cm
	float total_volume = 0.f; //in cm

#ifdef visualize	
	pcl::visualization::PCLVisualizer viewer0("ICP demo");


	int v1(0);

	int v2(1);

	viewer0.createViewPort(0.0, 0.0, 0.5, 1.0, v1);

	viewer0.createViewPort(0.5, 0.0, 1.0, 1.0, v2);
	
	viewer0.addPointCloud(cloud, to_string(cv::getTickCount()), v1);
	

#endif	
	int a, b, c;
	uint32_t rgb;
	srand(time(NULL));
		
	for (int i=0; i< slices_N; i++){
		// Create the filtering object
		
		//get cloud section
		pcl::PassThrough<PointT> pass;
		pass.setInputCloud (cloud);
		pass.setFilterFieldName ("z");
		pass.setFilterLimits (min_pt[2] + i*slice_height, min_pt[2] + (i+1)*slice_height);
		//pass.setFilterLimitsNegative (true);
		pass.filter (*cloud_filtered);
		
		if (cloud_filtered->size()<10)
			continue;
			
		//project cloud_filtered to z = 1 plane
		pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
		coefficients->values.resize(4);
		coefficients->values[0] = coefficients->values[1] = 0;
		coefficients->values[2] = 1.0;
		coefficients->values[3] = 0;

		// Create the filtering object
		pcl::ProjectInliers<PointT> proj;
		proj.setModelType(pcl::SACMODEL_PLANE);
		proj.setInputCloud(cloud_filtered);
		proj.setModelCoefficients(coefficients);
		proj.filter(*cloud_projected);
		
		slice_volume = getMeshArea(cloud_projected)*slice_height*pow(100.f,3);
		total_volume += slice_volume;

		a = (double)(rand() % 256);
		b = (double)(rand() % 256);
		c = (double)(rand() % 256);

		rgb = a << 16 | b << 8 | c;
		//cout<<"a: "<<a<<" b: "<<b<<" c: "<<c<<endl;

		for (auto &j : cloud_filtered->points){
			j.rgb = *reinterpret_cast<float*> (&rgb); 
			colored_cloud->points.push_back(j);
		}
	
	}
#ifdef visualize	
	std::stringstream file_name;
	file_name << current_pot_id << "_" << current_date << ".pcd";

	//save pcd file
	colored_cloud->width = 1;
	colored_cloud->height = colored_cloud->points.size();
	pcl::io::savePCDFile(file_name.str(), *colored_cloud);
	viewer0.addPointCloud<PointT>(colored_cloud, to_string(cv::getTickCount()), v2);	
	//viewer0.addCoordinateSystem (1.0);
	viewer0.spin();
#endif
	
	cout<<"total_sliced_volume: "<<total_volume<<endl;

	return total_volume;
}

float getPercentileHeight(PointCloudT::Ptr cloud){
		
	Eigen::Vector4f min_pt, max_pt;
	pcl::getMinMax3D(*cloud, min_pt, max_pt);
	
	cout<<"max_pt[2]: "<<max_pt[2] << " min_pt[2]: "<<min_pt[2]<<endl;
	
	float height_ori = max_pt[2] - min_pt[2];
	
	cout<<"height_ori: "<<height_ori<<endl;
	
	//float arr[] = {};
	std::vector< float > arr;
	
	PointT p;
    for (int i=0; i<cloud->size();i++){
    	p = cloud->points[i];
		arr.push_back(p.z);
    }
	
	sort(arr.begin(), arr.end()); 
 
    //cout << "\nArray after sorting using "
            //"default sort is : \n";
    //for (int i = 0; i < arr.size(); ++i)
        //cout << arr[i] << " ";
      
    cout<<"arr size "<<arr.size()<<endl;
    
    float percentile = 0.95f;
    float low_idx = (arr.size()-1)*(1.f-percentile);
    float high_idx = (arr.size()-1)*percentile;
    cout<<"low_idx: "<<low_idx<< " high_idx: "<<high_idx<<endl;
    
    float low_z = arr[int(low_idx)];
    float high_z = arr[int(high_idx)];
    float percentile_height = high_z - low_z;
    
    cout<<"low_z: "<<low_z<< " high_z: "<<high_z<<endl;
    cout<<"percentile_height: "<<percentile_height<<endl;
        
    return 0.f;

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

		//if (i < 50) {
		//	cout << "i: " << i << ", sd: " << point_sd << "point_indices size: " << point_indices->indices.size() << endl;
		//}
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

void holisticMeasurement(PointCloudT::Ptr cloud) {

	pcl::MomentOfInertiaEstimation<PointXYZ> feature_extractor;
	pcl::PointXYZ min_point_OBB;
	pcl::PointXYZ max_point_OBB;
	pcl::PointXYZ position_OBB;
	Eigen::Matrix3f rotational_matrix_OBB;
	float major_value, middle_value, minor_value;
	Eigen::Vector3f major_vector, middle_vector, minor_vector;

	ofstream outFile;
	Rosette ros;

	//ros.volume = getVoxelVolume(cloud);
	ros.volume = getSlicedVolume(cloud);
	
	PointCloud<PointXYZ>::Ptr temp(new PointCloud<PointXYZ>());

	Eigen::Vector4f min_pt, max_pt;

	//ros.micro_curvature = getMicroCurvature(cloud); //too slow

	ros.num_size = cloud->size();

	pcl::getMinMax3D(*cloud, min_pt, max_pt);
	
	//cout<<"min_pt: \n"<<min_pt<<endl;
	//cout<<"max_pt: \n"<<max_pt<<endl;

	ros.plant_height = (max_pt(2) - min_pt(2))*100.f;
	
	//getPercentileHeight(cloud);

	pcl::copyPointCloud(*cloud, *temp);
	feature_extractor.setInputCloud(temp);
	feature_extractor.compute();

	feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);
	feature_extractor.getEigenValues(major_value, middle_value, minor_value);
	//feature_extractor.getEigenVectors(major_vector, middle_vector, minor_vector);

	ros.holistic_curvature = minor_value / (minor_value + middle_value + major_value);


	//cout << "major_value: " << major_value << ", middle_value: " << middle_value << ", minor_value: " << minor_value << endl;	
	//cout << "major_vector: " << major_vector << endl;
	//cout << "middle_vector: " << middle_vector << endl;
	//cout << "minor_vector: " << minor_vector << endl;

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

	ros.length_of_boundingbox = dis14*100.f;
	ros.width_of_boundingbox = dis15*100.f;
	ros.height_of_boundingbox = dis12*100.f;
	ros.boundingbox_volume = dis12*dis14*dis15*pow(100.f, 3);
	ros.holistic_aspect_ratio = max(max(dis14, dis15), dis12) / min(min(dis14, dis15), dis12);

	ConvexHull<PointT> chull;
	PointCloudT::Ptr cloud_chull(new PointCloudT);
	chull.setInputCloud(cloud);
	chull.setComputeAreaVolume(true);
	chull.reconstruct(*cloud_chull);
	ros.convex_area = chull.getTotalArea() * pow(100.f, 2);
	ros.convex_hull_volume = chull.getTotalVolume()*pow(100.f, 3);

	ros.area = getMeshArea(cloud)* pow(100.f, 2);

	//ros.holistic_solidity = ros.convex_hull_volume / ros.boundingbox_volume;
	ros.holistic_solidity = ros.volume / ros.convex_hull_volume;

	ros.holistic_area_convexity = ros.area / ros.convex_area;

	ros.area_based_volume = ros.area*leave_thickness*100.f;
	ros.area_based_solidity = ros.area_based_volume/ros.convex_hull_volume;

	//ros.holistic_solidity
#ifdef visualize
	cout << "area: " << ros.area << ", ros.convex_area: " << ros.convex_area << ", ros.holistic_area_convexity: " << ros.holistic_area_convexity<< endl;
	
	cout << "boundingbox_volume: " << ros.boundingbox_volume << ", ros.convex_hull_volume: " << ros.convex_hull_volume << endl;
	
	cout <<"area_volume: "<< ros.area_based_volume << " area_based_solidity: "<< ros.area_based_solidity <<endl; //suppose 5mm 厚
	
	cout<<"slice volume: "<<ros.volume << " sliced_volume/convexhull_volume " << ros.volume/ros.convex_hull_volume <<endl;
	
	pcl::visualization::PCLVisualizer viewer("demo");

	int v1(0);
	int v2(1);
	viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);
	viewer.createViewPort(0.5, 0.0, 1.0, 1.0, v2);

	viewer.registerPointPickingCallback(&pp_callback);

	viewer.addPointCloud(cloud, to_string(cv::getTickCount()), v1);

	pcl::visualization::PointCloudColorHandlerCustom<PointT> cloud_target_color_h(cloud_chull, 20, 20, 180);
	viewer.addPointCloud(cloud_chull, cloud_target_color_h, to_string(cv::getTickCount()), v2);
	  
	PolygonMesh mesh;
	chull.reconstruct (mesh);
	
	viewer.addPolygonMesh(mesh,"meshes",v2);
	//viewer.setShapeRenderingProperties ( pcl::visualization::PCL_VISUALIZER_OPACITY, 0.5, "meshes", v2 );

	viewer.spin();
	
#endif

#ifndef visualize
	outFile.open(holistic_measurement_path, ios::app);

	if (measure_file_first_line) {
		outFile << "pot_id,sample_date,";
		outFile << "num_size_3d,plant_height_3d,holistic_aspect_ratio_3d,";
		outFile << "length_of_boundingbox_3d,width_of_boundingbox_3d,height_of_boundingbox_3d,";
		outFile << "area_3d,convex_area_3d,holistic_area_convexity_3d,";
		outFile << "convex_hull_volume_3d,boundingbox_volume_3d,volume_3d,solidity_3d,";
		//outFile << "area_based_volume_3d,area_based_solidity_3d,";
		outFile << "holistic_curvature_3d\n";
		measure_file_first_line = false;
	}

	outFile << current_pot_id << "," << current_date << ",";
	outFile << ros.num_size << "," << ros.plant_height << "," << ros.holistic_aspect_ratio << ",";
	outFile << ros.length_of_boundingbox << "," << ros.width_of_boundingbox << "," << ros.height_of_boundingbox << ",";
	outFile << ros.area << "," << ros.convex_area << "," << ros.holistic_area_convexity << ",";
	outFile << ros.convex_hull_volume << "," << ros.boundingbox_volume << "," << ros.volume <<","<<ros.holistic_solidity << ",";
	//outFile << ros.area_based_volume << "," << ros.area_based_solidity <<",";
	outFile << ros.holistic_curvature << "," << "\n";

	outFile.close();

#endif


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

			if (cluster_indices.size() < 1)
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

			if (dist < 0.005f) { 
		
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

		pcl::PointXYZ pt1(p1(0), p1(1), p1(2));
		pcl::PointXYZ pt2(p2(0), p2(1), p2(2));
		pcl::PointXYZ pt4(p4(0), p4(1), p4(2));
		pcl::PointXYZ pt5(p5(0), p5(1), p5(2));

		float dis12 = euclideanDistance(pt1, pt2);
		float dis14 = euclideanDistance(pt1, pt4);
		float dis15 = euclideanDistance(pt1, pt5);

		lf.leaf_length = dis14*100.f;
		lf.leaf_width = dis15*100.f;  
	
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
			for (int i = 1;i < 2;i++) { 
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
		viewer0.addLine (pt1, pt2, 1.0, 0.0, 0.0, to_string(cv::getTickCount()), v1);
		viewer0.addLine (pt1, pt4, 1.0, 0.0, 0.0, to_string(cv::getTickCount()), v1);
		viewer0.addLine (pt1, pt5, 1.0, 0.0, 0.0, to_string(cv::getTickCount()), v1);


#endif
	}



#ifdef visualize
	viewer0.spin();
#endif

}
