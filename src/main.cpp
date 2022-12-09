#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/navigation/RadarFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/slam/dataset.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/dataset.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/inference/Symbol.h>
#include <tf/tf.h>
#include <tf/transform_datatypes.h>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <x86_64-linux-gnu/sys/stat.h>

#define USE_COMBINED

using namespace std;
using namespace gtsam;
using symbol_shorthand::B; // 陀螺仪残差  (ax,ay,az,gx,gy,gz)
using symbol_shorthand::V; // 用表示      速度导数(xdot,ydot,zdot)
using symbol_shorthand::X; // 用作表示    姿态(x,y,z,r,p,y)
string inputfile = "/home/invoker/code/gtsam_test/data/data_suzhouluce/1113_luce01.txt";
string outputfile = "/home/invoker/code/gtsam_test/results/data_suzhouluce/imuradaricp_result.csv";
PreintegrationType *imu_preintegrated_;
Values result;
//    Matrix3 biasAccCovariance;     3*3矩阵 加速度计的协防差矩阵，（可以根据残差计算加速度雅克比矩阵逐步更新）
//    Matrix3 biasOmegaCovariance;   3*3矩阵 陀螺仪的协防差矩阵， （可以根据残差计算雅克比矩阵递归更新预计分值，防止从头计算）
//    Matrix6 biasAccOmegaInt;       6*6矩阵 位置残关于加速度和速度残差的协防差，用作更新预计分

static double azimuth2heading(double azimuth)
{
    if (azimuth < 180 && azimuth > 0)
    {
        return -azimuth;
    }
    else if (azimuth > 180 && azimuth < 360)
    {
        return 360 - azimuth;
    }
    else
    {
        std::cout << "wrong azimuth!" << endl;
    }
}

static double degree2rad(double theta)
{
    return theta * M_PI / 180.0;
}

static Eigen::Vector3d R2rpy(const Eigen::Matrix3d &R)
{
    Eigen::Vector3d n = R.col(0);
    Eigen::Vector3d o = R.col(1);
    Eigen::Vector3d a = R.col(2);

    Eigen::Vector3d rpy(3);
    double y = atan2(n(1), n(0));
    double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
    double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
    rpy(0) = r;
    rpy(1) = p;
    rpy(2) = y;

    return rpy / M_PI * 180.0;
}

int main(int argc, const char **argv)
{
    string result_path = "/home/invoker/code/gtsam_test/results/data_suzhouluce";

    if (access(result_path.c_str(), 0))
    {
        std::cout << "folder dose not exist!! Will create a new one!" << std::endl;
        std::string commmand = "mkdir " + result_path;
        system(commmand.c_str());
    }
    FILE *fp = fopen(outputfile.c_str(), "w+");
    // 输出
    fprintf(fp, "#time(s),x(m),y(m),z(m),roll,pitch,yaw,gt_x(m),gt_y(m),gt_z(m),roll,pitch,yaw,bias_gx,bias_gy,bias_gz,bias_wx,bias_wy,bias_wz\n");

    // 解析 CSV
    ifstream file(inputfile.c_str());
    string value;

    Eigen::Matrix<double, 10, 1> initial_state = Eigen::Matrix<double, 10, 1>::Zero();
    // N，E，D，qx,qy,qz,qw,VelN,VelE,VelD
    // getline(file, value, ',');
    // for (int i = 0; i < 9; i++)
    // {
    //     getline(file, value, ',');
    //     initial_state(i) = atof(value.c_str()); //转为浮点型
    // }
    // getline(file, value, '\n'); //换行
    // initial_state(9) = atof(value.c_str());
    // getline(file, value);//read first line
    // Vector3 prior_euler(0.7192, -0.8241, 269.8044);//0s
    // Vector3 prior_euler(1.409, 3.79099989, 90.0); // data_suzhou_1s
    Vector3 prior_euler(1.832, 4.144, 6.011); // data_suzhou_1113luce01
    // Vector3 prior_euler(2.5708, 0.1559301, -168.0908);
    Rot3 prior_rotation = Rot3::ypr(degree2rad(prior_euler(0)),
                                    degree2rad(prior_euler(1)),
                                    degree2rad(prior_euler(2)));
    // Point3 prior_point(0.0, 7.2411, -0.1);// data_suzhou_1s
    Point3 prior_point(11.31134, 0, -0.079999);// data_suzhou_1113luce01
    // Point3 prior_point(-0.1223, -0.0028506, 0.008); // data_our_1s
    Quaternion temp = prior_rotation.toQuaternion();
    Pose3 prior_pose(prior_rotation, prior_point);             // 初始位姿
    // Vector3 prior_velocity(0.28420691, 0.0001247, 0.00821895); // data_our初始速度
    // Vector3 prior_velocity(0.238, 7.418, -0.238); //data_suhzou初始速度
    Vector3 prior_velocity(11.23599, -0.02244689, -0.29066732); //data_suzhou_1113luce01初始速度
    imuBias::ConstantBias prior_imu_bias; // 残差，默认设为0
    NavState prior_nav(prior_pose, prior_velocity);

    // 初始化值
    Values initial_values;
    int correction_count = 0;
    int icp_correction_count = 0;
    // 位姿
    initial_values.insert(X(correction_count), prior_pose);
    // 速度
    initial_values.insert(V(correction_count), prior_velocity);
    // 残差
    initial_values.insert(B(correction_count), prior_imu_bias);
    cout << "initial state:\n"
         << initial_state.transpose() << endl;
    // 设置噪声模型
    // 一般为设置为对角噪声
    Vector6 pose_noise_sigma;
    pose_noise_sigma << 0.01, 0.01, 0.01, 0.1, 0.1, 0.1;
    noiseModel::Diagonal::shared_ptr pose_noise_model = noiseModel::Diagonal::Sigmas(pose_noise_sigma);
    noiseModel::Diagonal::shared_ptr velocity_noise_model = noiseModel::Isotropic::Sigma(3, 0.1);
    noiseModel::Diagonal::shared_ptr bias_noise_model = noiseModel::Isotropic::Sigma(6, 0.001);
    NonlinearFactorGraph *graph = new NonlinearFactorGraph();
    graph->add(PriorFactor<Pose3>(X(correction_count), prior_pose, pose_noise_model));
    graph->add(PriorFactor<Vector3>(V(correction_count), prior_velocity, velocity_noise_model));
    graph->add(PriorFactor<imuBias::ConstantBias>(B(correction_count), prior_imu_bias, bias_noise_model));
    // 使用传感器信息构建IMU的噪声模型
    double accel_noise_sigma = 0.0005886;
    double gyro_noise_sigma = 0.0001744444;
    double accel_bias_rw_sigma = 0.0000104050763;
    double gyro_bias_rw_sigma = 0.00000342641249;

    Matrix33 measured_acc_cov = Matrix33::Identity(3, 3) * pow(accel_noise_sigma, 2);
    Matrix33 measured_omega_cov = Matrix33::Identity(3, 3) * pow(gyro_bias_rw_sigma, 2);
    Matrix33 integration_error_cov = Matrix33::Identity(3, 3) * 1e-8; // 速度积分误差
    Matrix33 bias_acc_cov = Matrix33::Identity(3, 3) * pow(accel_bias_rw_sigma, 2);
    Matrix33 bias_omega_cov = Matrix33::Identity(3, 3) * pow(gyro_bias_rw_sigma, 2);
    Matrix66 bias_acc_omega_int = Matrix66::Identity(6, 6) * 1e-5; // 积分骗到误差

    boost::shared_ptr<PreintegratedCombinedMeasurements::Params> p = PreintegratedCombinedMeasurements::Params::MakeSharedD();
    // MakeSharedD:NED坐标系，g默认为 9.81，这里设置为0
    // MakeSharedU：NEU坐标系，g默认为 9.81

    // 设置预积分分参数
    p->accelerometerCovariance = measured_acc_cov;
    p->integrationCovariance = integration_error_cov;
    p->gyroscopeCovariance = measured_omega_cov;

    // 预计分测量值
    p->biasAccCovariance = bias_acc_cov;
    p->biasAccOmegaInt = bias_acc_omega_int;
    p->biasOmegaCovariance = bias_omega_cov;
#ifdef USE_COMBINED
    imu_preintegrated_ = new PreintegratedCombinedMeasurements(p, prior_imu_bias);
#else
    imu_preintegrated_ = new PreintegratedImuMeasurements(p, prior_imu_bias);
#endif
    // 保存上一次的imu积分值和结果
    NavState prev_state(prior_pose, prior_velocity);
    NavState prop_state = prev_state;
    imuBias::ConstantBias prev_bias = prior_imu_bias; //

    // 记录总体误差
    double current_position_error = 0.0, current_orientation_error = 0.0;

    double output_time = 0;
    double dt = 0.01; // 积分时间

    // 使用数据进行迭代
    while (file.good())
    {
        getline(file, value, ' ');
        int type = atoi(value.c_str()); // 字符转为整形
        if (type == 0)
        { // IMU 测量数据
            Eigen::Matrix<double, 6, 1> imu = Eigen::Matrix<double, 6, 1>::Zero();
            // 读取imu数据
            for (int i = 0; i < 5; ++i)
            {
                getline(file, value, ' ');
                imu(i) = atof(value.c_str());
            }
            getline(file, value, '\n');
            imu(5) = atof(value.c_str());
            // just for test
            // imu(0) = imu(0) + 5;

            // 检测测量值加入预计分
            imu_preintegrated_->integrateMeasurement(imu.head<3>(), imu.tail<3>(), dt);
        }
        else if (type == 1)
        {
            // velocity
            Eigen::Vector3d vel_esti = Eigen::Vector3d::Zero();
            for (int i = 0; i < 2; i++)
            {
                getline(file, value, ' ');
                vel_esti(i) = atof(value.c_str());
            }
            getline(file, value, '\n');
            vel_esti(2) = atof(value.c_str());
            vel_esti(2) = 0.0;
            correction_count++;

            Vector3 vel_noise_sigma;
            vel_noise_sigma << 0.1, 0.1, 0.3;
            noiseModel::Diagonal::shared_ptr radar_noise_model = noiseModel::Diagonal::Sigmas(vel_noise_sigma);
            RadarFactor radar_factor(X(correction_count), V(correction_count),
                                     vel_esti,
                                     radar_noise_model);

            // Eigen::Matrix3d bRn = prev_state.R();
            // Vector3 Vned = bRn * vel_esti;
            // RadarFactor radar_factor(V(correction_count),
            //                          Vned,
            //                          radar_noise_model);

            graph->add(radar_factor);

#ifdef USE_COMBINED
            // 预计分测量值
            PreintegratedCombinedMeasurements *preint_imu_combined = dynamic_cast<PreintegratedCombinedMeasurements *>(imu_preintegrated_);
            // IMU 因子
            // typedef NoiseModelFactor6<Pose3, Vector3, Pose3, Vector3,imuBias::ConstantBias, imuBias::ConstantBias>
            CombinedImuFactor imu_factor(X(correction_count - 1), V(correction_count - 1),
                                         X(correction_count), V(correction_count),
                                         B(correction_count - 1), B(correction_count),
                                         *preint_imu_combined);
            graph->add(imu_factor);
#else
            // 预计分测量值
            PreintegratedImuMeasurements *preint_imu = dynamic_cast<PreintegratedImuMeasurements *>(imu_preintegrated_);
            ImuFactor imu_factor(X(correction_count - 1), V(correction_count - 1),
                                 X(correction_count), V(correction_count),
                                 B(correction_count - 1),
                                 *preint_imu);
            graph->add(imu_factor);
            imuBias::ConstantBias zero_bias(Vector3(0, 0, 0), Vector3(0, 0, 0));
            graph->add(BetweenFactor<imuBias::ConstantBias>(B(correction_count - 1),
                                                            B(correction_count),
                                                            zero_bias, bias_noise_model));
#endif

            // 迭代更新求解imu预测值
            prop_state = imu_preintegrated_->predict(prev_state, prev_bias);
            Eigen::Vector3d curr_euler = R2rpy(prop_state.pose().rotation().toQuaternion().toRotationMatrix());

            initial_values.insert(X(correction_count), prop_state.pose());
            initial_values.insert(V(correction_count), prop_state.v());
            initial_values.insert(B(correction_count), prev_bias);
        }
        else if (type == 2)
        {
            // icp
            Eigen::Matrix<double, 4, 4> icp_Trans = Eigen::Matrix<double, 4, 4>::Zero();
            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    getline(file, value, ' ');
                    icp_Trans(i, j) = atof(value.c_str());
                }
            }
            getline(file, value, '\n');
            Eigen::Matrix3d icp_Rtrans = icp_Trans.topLeftCorner<3, 3>();
            Eigen::Vector3d icp_Ptrans = icp_Trans.topRightCorner<3, 1>();
            Rot3 icp_R(icp_Rtrans);
            Pose3 icp_pose(icp_R, icp_Ptrans);

            Vector6 icp_noise_sigma;
            icp_noise_sigma << 0.001, 0.001, 0.001, 0.05, 0.05, 0.05;
            noiseModel::Diagonal::shared_ptr icp_noise_model = noiseModel::Diagonal::Sigmas(icp_noise_sigma);
            graph->add(BetweenFactor<Pose3>(X(correction_count - 1),
                                            X(correction_count),
                                            icp_pose, icp_noise_model));
        }
        else if (type == 3)
        { // Gps测量数据
            Eigen::Matrix<double, 6, 1> gps = Eigen::Matrix<double, 6, 1>::Zero();
            for (int i = 0; i < 5; i++)
            {
                getline(file, value, ' ');
                gps(i) = atof(value.c_str());
            }
            getline(file, value, '\n');
            gps(5) = atof(value.c_str());

            noiseModel::Diagonal::shared_ptr gps_correction_noise = noiseModel::Isotropic::Sigma(3, 0.1);
            GPSFactor gps_factor(X(correction_count),
                                 Point3(gps(0), gps(1), gps(2)), //(N,E,D)
                                 gps_correction_noise);
            // graph->add(gps_factor);

            // 求解
            LevenbergMarquardtOptimizer optimizer(*graph, initial_values);
            result = optimizer.optimize();

            // 更新下一步预计分初始值
            // 导航状态
            prev_state = NavState(result.at<Pose3>(X(correction_count)),
                                  result.at<Vector3>(V(correction_count)));
            // 偏导数
            prev_bias = result.at<imuBias::ConstantBias>(B(correction_count));
            // 更新预计分值
            imu_preintegrated_->resetIntegrationAndSetBias(prev_bias);

            // 计算角度误差和误差
            Vector3 gtsam_position = prev_state.pose().translation();
            // 位置误差
            Vector3 position_error = gtsam_position - gps.head<3>();
            // 误差的范数
            current_position_error = position_error.norm(); // 归一化

            // 姿态误差
            Quaternion gtsam_quat = prev_state.pose().rotation().toQuaternion();
            Quaternion gps_quat = Rot3::ypr(degree2rad(gps(3)), degree2rad(gps(4)), degree2rad(gps(5))).toQuaternion();
            Quaternion quat_error = gtsam_quat * gps_quat.inverse();
            quat_error.normalized();                                                               // 归一化
            Vector3 euler_angle_error(quat_error.x() * 2, quat_error.y() * 2, quat_error.z() * 2); // 转换为欧拉角误差
            current_orientation_error = euler_angle_error.norm();

            Eigen::Vector3d gps_euler(gps(3), gps(4), gps(5));
            Eigen::Vector3d gtsam_euler = R2rpy(gtsam_quat.toRotationMatrix());
            Eigen::Vector3d gt_euler = R2rpy(gps_quat.toRotationMatrix());
            // 输出误差
            cout << "Position error:" << current_position_error << "\t "
                 << "Angular error:" << current_orientation_error << "\n";
            fprintf(fp, "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n",
                    output_time, gtsam_position(0), gtsam_position(1), gtsam_position(2),
                    gtsam_euler(0), gtsam_euler(1), gtsam_euler(2),
                    gps(0), gps(1), gps(2),
                    gps_euler(0), gps_euler(1), gps_euler(2),
                    prev_bias.vector()(0), prev_bias.vector()(1), prev_bias.vector()(2),
                    prev_bias.vector()(3), prev_bias.vector()(4), prev_bias.vector()(5));

            output_time += 1.0;
        }
        else
        {
            cerr << "ERROR parsing file\n";
            return 1;
        }
    }
    fclose(fp);
    cout << "完成,结果见：" << outputfile << endl;
    return 0;
}
