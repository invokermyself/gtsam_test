#include <gtsam/geometry/Pose2.h>
// Pose2:(x,y,theta)
// Point2:(x,y)
#include <gtsam/geometry/Point2.h>
//每个元必须key，在gtsam中
//可以使用整数int，或者symbol自定义key
#include <gtsam/inference/Symbol.h>
// PriorFactor           先验因子(初始化起点)
// BetweenFactor         两个因子之间
// BearingRangeFactor    gtsam中用来确定距离方位（路标点landmark）的因子
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/sam/BearingRangeFactor.h>
//信任区域法
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
//协方差
#include <gtsam/nonlinear/Marginals.h>
//获取值
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <fstream>
using namespace std;
using namespace gtsam;

int main()
{
    // 声明一个优化图和一些变量
    NonlinearFactorGraph graph;
    Symbol x1('x', 1), x2('x', 2), x3('x', 3);
    Symbol l1('l', 1), l2('l', 2);

    // 起点
    Pose2 prior(0.0, 0.0, 0.0);
    noiseModel::Diagonal::shared_ptr priorNoise = noiseModel::Diagonal::Sigmas(Vector3(0.2, 0.0, 0.0));
    graph.emplace_shared<PriorFactor<Pose2>>(x1, prior, priorNoise);

    // 添加两个里程计
    Pose2 odometry1(2.0, 0.0, 0.0);
    noiseModel::Diagonal::shared_ptr odoNoise = noiseModel::Diagonal::Sigmas(Vector3(0.2, 0.2, 0.1));
    graph.emplace_shared<BetweenFactor<Pose2>>(x1, x2, odometry1, odoNoise);
    Pose2 odometry2(1.8, 0.0, 0.0);
    graph.emplace_shared<BetweenFactor<Pose2>>(x2, x3, odometry2, odoNoise);

    // 地图点
    //添加Range-Bearing 到两俩个不同的landmarks
    noiseModel::Diagonal::shared_ptr measurementNoise = noiseModel::Diagonal::Sigmas(Vector2(0.1, 0.2));
    Rot2 bearing11 = Rot2::fromDegrees(45); // l1,x1,x2,构成的夹角
    Rot2 bearing21 = Rot2::fromDegrees(90); // x1,x2,l1,构成的夹角
    Rot2 bearing32 = Rot2::fromDegrees(90); // x1,x2,l2,构成的夹角
    double range11 = sqrt(8), range21 = 2.0, range32 = 2.0;

    //地图点和位姿之间的约束
    graph.emplace_shared<BearingRangeFactor<Pose2, Point2>>(x1, l1, bearing11, range11, measurementNoise);
    graph.emplace_shared<BearingRangeFactor<Pose2, Point2>>(x2, l1, bearing21, range21, measurementNoise);
    graph.emplace_shared<BearingRangeFactor<Pose2, Point2>>(x3, l2, bearing32, range32, measurementNoise);

    //初始值
    Values initialEstimate;
    initialEstimate.insert(x1, Pose2(0.5, 0.0, 0.2));
    initialEstimate.insert(x2, Pose2(2.3, 0.1, -0.2));
    initialEstimate.insert(x3, Pose2(4.1, 0.1, 0.1));
    initialEstimate.insert(l1, Point2(1.8, 2.1));
    initialEstimate.insert(l2, Point2(4.1, 1.8));

    //求解
    LevenbergMarquardtOptimizer optimizer(graph, initialEstimate);
    Values result = optimizer.optimize();
    //优化后的结果
    result.print("Final Result:\n");
    Marginals marginals(graph, result);
    cout << "marginals x1:" << endl
         << marginals.marginalCovariance(x1) << endl;
    cout << "marginals x2:" << endl
         << marginals.marginalCovariance(x2) << endl;
    cout << "marginals x3:" << endl
         << marginals.marginalCovariance(x3) << endl;
    cout << "marginals l1:" << endl
         << marginals.marginalCovariance(l1) << endl;
    cout << "marginals l2:" << endl
         << marginals.marginalCovariance(l2) << endl;
}
