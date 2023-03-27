#include <igl/opengl/glfw/Viewer.h>
#include <igl/readOFF.h>
#include <igl/readOBJ.h>
#include <igl/project.h>
#include <iostream>
#include <ostream>
#include <igl/principal_curvature.h>
#include <igl/per_vertex_normals.h>
#include <igl/jet.h>
#include <igl/avg_edge_length.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/gaussian_curvature.h>
#include<igl/ray_mesh_intersect.h>
#if defined(_WIN32) || defined(WIN32) 
#define _USE_MATH_DEFINES
#include <math.h>
#endif

using namespace Eigen; // to use the classes provided by Eigen library
using namespace std;

MatrixXd V1; // matrix storing vertex coordinates of the input mesh (n rows, 3 columns)
MatrixXi F1; // incidence relations between faces and edges (f columns)
Eigen::MatrixXd N; //Normals
Eigen::MatrixXd triangle_normals;
Eigen::MatrixXd PD1, PD2; // #V by 3 matrices of maximal and minimal curvature directions
Eigen::VectorXd PV1, PV2; // #V by 1 vectors of maximal and minimal curvature values
// Compute vertex-triangle adjacency
std::vector<std::vector<int>> VF;
std::vector<std::vector<int>> VFi;
double threshold = 1.5;
double radius_scale = 2.5+1.;
static void largest_eig_2x2(double m1, double m12, double m2, Eigen::Vector2d &t1, double &q1)
{
    q1 = 0.5 * (m1 + m2);
    if (q1 > 0.0)
        q1 += sqrt(pow(m12,2) + 0.25 * pow((m2-m1),2));
    else
        q1 -= sqrt(pow(m12,2) + 0.25 * pow((m2-m1),2));

    // Find corresponding eigenvector
    t1 = Vector2d(m2 - q1, -m12);
    t1.normalized();
}

void compute_viewdependent_curvature(const MatrixXd V, int i, double ndotv,
			  double u2, double uv, double v2,
			  double &q1, Vector2d &t1)
{
	// Find the entries in Q = S * P^-1
	//                       = S + (sec theta - 1) * S * w * w^T
	double sectheta_minus1 = 1.0 / fabs(ndotv) - 1.0;
	double Q11 = PV1(i) * (1.0 + sectheta_minus1 * u2);
	double Q12 = PV1(i) * (      sectheta_minus1 * uv);
	double Q21 = PV2(i) * (      sectheta_minus1 * uv);
	double Q22 = PV2(i) * (1.0 + sectheta_minus1 * v2);

	// Find the three entries in the (symmetric) matrix Q^T Q
	double QTQ1  = Q11 * Q11 + Q21 * Q21;
	double QTQ12 = Q11 * Q12 + Q21 * Q22;
	double QTQ2  = Q12 * Q12 + Q22 * Q22;

	// Compute eigenstuff
	largest_eig_2x2(QTQ1, QTQ12, QTQ2, t1, q1);
}


// Compute D_{t_1} q_1 - the derivative of max view-dependent curvature
// in the principal max view-dependent curvature direction.
void compute_Dt1q1(const MatrixXd V, int i, double ndotv,
		   const vector<double> &q1, const vector<Vector2d> &t1,
		   double &Dt1q1)
{
	const RowVector3d &v0 = V.row(i);
	double this_viewdep_curv = q1[i];
	RowVector3d world_t1 = t1[i](0) * PD1.row(i) + t1[i](1) * PD2.row(i);
	RowVector3d world_t2 = RowVector3d(N.row(i)).cross(world_t1);
	double v0_dot_t2 = v0.dot(world_t2);

	Dt1q1 = 0.0;
	int n = 0;

	int naf = VF[i].size();
	for (int j = 0; j < naf; j++) {
		int f = VF[i][j];
		int i1,i2;
		if(RowVector3d(V.row(F1(f,0))) == v0 )
		{
			i1 = F1(f,1);
			i2 = F1(f,2);
		}
		else if(RowVector3d(V.row(F1(f,1))) == v0 )
		{
			i1 = F1(f,0);
			i2 = F1(f,2);
		}
		else if(RowVector3d(V.row(F1(f,2))) == v0 )
		{
			i1 = F1(f,0);
			i2 = F1(f,1);
		}

		const RowVector3d &v1 = V.row(i1);
		const RowVector3d &v2 = V.row(i2);

		double v1_dot_t2 = v1.dot(world_t2);
		double v2_dot_t2 = v2.dot(world_t2);
		double w1 = (v2_dot_t2 - v0_dot_t2) / (v2_dot_t2 - v1_dot_t2);

		if (w1 <= 0.0 || w1 >= 1.0)
			continue;
		
		double w2 = 1.0 - w1;
		RowVector3d p = w1 * v1 + w2 * v2;

		double interp_viewdep_curv = w1 * q1[i1] + w2 * q1[i2];

		double proj_dist = (p - v0).dot(world_t1);
		proj_dist *= fabs(ndotv);
		Dt1q1 += (interp_viewdep_curv - this_viewdep_curv) / proj_dist;
		n++;

		if (n == 2) {
			Dt1q1 *= 0.5;
			return;
		}
	}
}
vector<Vector3d> proj1;
vector<Vector3d> proj2;
void draw_segment_app_ridge(int v0, int v1, int v2,
			    double emax0, double emax1, double emax2,
			    double kmax0, double kmax1, double kmax2,
			    const RowVector3d &tmax0, const RowVector3d &tmax1, const RowVector3d &tmax2, igl::opengl::glfw::Viewer& viewer,bool to_center)
{
	
	double w10 = fabs(emax0) / (fabs(emax0) + fabs(emax1));
	double w01 = 1.0 - w10;
	RowVector3d p01 = w01 * V1.row(v0) + w10 * V1.row(v1);
	double k01 = fabs(w01 * kmax0 + w10 * kmax1);
	RowVector3d p12;
	double k12;
	if (to_center) {//center 
		p12 = (V1.row(v0) + V1.row(v1) + V1.row(v2)) / 3.0;
		k12 = fabs(kmax0 + kmax1 + kmax2) / 3.0;
	} else 
	{
		double w21 = fabs(emax1) / (fabs(emax1) + fabs(emax2));
		double w12 = 1.0 - w21;
		p12 = w12 * V1.row(v1) + w21 * V1.row(v2);
		k12 = (w12 * kmax1 + w21 * kmax2);
	}
	k01 -= threshold;
	if (k01 < 0.0)
		k01 = 0.0;
	k12 -= threshold;
	if (k12 < 0.0)
		k12 = 0.0;
	if (k01 == 0.0 && k12 == 0.0)
		return;

	{
		RowVector3d v1v0 = RowVector3d(V1.row(v1)-V1.row(v0));
		RowVector3d v2v1 = RowVector3d(V1.row(v2)-V1.row(v1));
		RowVector3d perp = 0.5*((v1v0.cross(v2v1))).cross(p01-p12);
		if ((tmax0.dot(perp) <= 0.0) ||
		    (tmax1.dot(perp) >= 0.0)||
		    (tmax2.dot(perp) <= 0.0) )
			return;
	}
	// Fade lines
	// if (draw_faded) {
		k01 /= (k01 + threshold);
		k12 /= (k12 + threshold);
	// } else
	 {
		// k01 = 1.0;
		// k12 = 1.0;
	 }
	 double fade = 1.0 - (k01+k12)*0.5;
	 if(fade <0.95)
	 fade-=0.6;
	 if(fade <0.)
	 fade = 0.0;
	// viewer.data().add_points(p01,RowVector3d(0,0,0));
	// viewer.data().add_points(p12,RowVector3d(0,0,0));
	proj1.push_back(p01);
	proj2.push_back(p12);
	viewer.data().add_edges (p01,p12,RowVector3d(fade,fade,fade));

}


// Draw apparent ridges in a triangle
void draw_face_app_ridges(RowVector3d camera_position,int v0, int v1, int v2,
			  const vector<double> &ndotv, const vector<double> &q1,
			  const vector<Vector2d> &t1, const vector<double> &Dt1q1, igl::opengl::glfw::Viewer& viewer)
{

// 	// Backface culling is turned off: getting contours from the
// 	// apparent ridge definition requires us to process faces that
// 	// may be (just barely) backfacing...
	
	if ((ndotv[v0] <= 0.0 && ndotv[v1] <= 0.0 && ndotv[v2] <= 0.0))
	{
		// if()
		// RowVector3d dir = V1.row(v0) - camera_position;
		// igl::Hit hit;
		// igl::ray_mesh_intersect(camera_position,dir,V1,F1,hit);
		// if(((V1.row(F1(hit.id,0))*(1-hit.u-hit.v)+V1.row(F1(hit.id,1))*hit.u+V1.row(F1(hit.id,2))*hit.v)- V1.row(v0)).norm() >(V1.row(v0)-V1.row(v1)).norm())
		// {
		// 	return;
		// }
		// dir = V1.row(v1) - camera_position;
		// igl::ray_mesh_intersect(camera_position,dir,V1,F1,hit);
		// if(((V1.row(F1(hit.id,0))*(1-hit.u-hit.v)+V1.row(F1(hit.id,1))*hit.u+V1.row(F1(hit.id,2))*hit.v)- V1.row(v1)).norm() >(V1.row(v0)-V1.row(v1)).norm())
		// {
		// 	return;
		// }
		// dir = V1.row(v2) - camera_position;
		// igl::ray_mesh_intersect(camera_position,dir,V1,F1,hit);
		// if(((V1.row(F1(hit.id,0))*(1-hit.u-hit.v)+V1.row(F1(hit.id,1))*hit.u+V1.row(F1(hit.id,2))*hit.v)- V1.row(v2)).norm() >(V1.row(v0)-V1.row(v2)).norm())
	return;
		
	}
	// Trivial reject if this face isn't getting past the threshold anyway
	const double &kmax0 = q1[v0];
	const double &kmax1 = q1[v1];
	const double &kmax2 = q1[v2];
	if (kmax0 <= threshold && kmax1 <= threshold && kmax2 <= threshold)
		return;

	// The "tmax" are the principal directions of view-dependent curvature,
	// flipped to point in the direction in which the curvature
	// is increasing.
	const double &emax0 = Dt1q1[v0];
	const double &emax1 = Dt1q1[v1];
	const double &emax2 = Dt1q1[v2];
	RowVector3d world_t1_0 = t1[v0](0) * PD1.row(v0) + t1[v0](1) * PD2.row(v0);
	RowVector3d world_t1_1 = t1[v1](0) * PD1.row(v1) + t1[v1](1) * PD2.row(v1);
	RowVector3d world_t1_2 = t1[v2](0) * PD1.row(v2) + t1[v2](1) * PD2.row(v2);
	RowVector3d tmax0 = Dt1q1[v0] * world_t1_0;
	RowVector3d tmax1 = Dt1q1[v1] * world_t1_1;
	RowVector3d tmax2 = Dt1q1[v2] * world_t1_2;
	// We have a "zero crossing" if the tmaxes along an edge
	// point in opposite directions
	bool z01 = ((tmax0.dot(tmax1)) <= 0.0);
	bool z12 = ((tmax1.dot(tmax2)) <= 0.0);
	bool z20 = ((tmax2.dot(tmax0)) <= 0.0);
	// if ((int)z01 + (int)z12 + (int)z20 < 2)
	// 	return;
	// Draw line segment
	if (!z01) {
		draw_segment_app_ridge(v1, v2, v0,
				       emax1, emax2, emax0,
				       kmax1, kmax2, kmax0,
				       tmax1, tmax2, tmax0, viewer,false);
	} else if (!z12) {
		draw_segment_app_ridge(v2, v0, v1,
				       emax2, emax0, emax1,
				       kmax2, kmax0, kmax1,
				       tmax2, tmax0, tmax1, viewer,false);
	} else if (!z20) {
		draw_segment_app_ridge(v0, v1, v2,
				       emax0, emax1, emax2,
				       kmax0, kmax1, kmax2,
				       tmax0, tmax1, tmax2, viewer,false);
	} 
	else {
		// All three edges have crossings -- connect all to center
		draw_segment_app_ridge(v1, v2, v0,
				       emax1, emax2, emax0,
				       kmax1, kmax2, kmax0,
				       tmax1, tmax2, tmax0, viewer,true);
		draw_segment_app_ridge(v2, v0, v1,
				       emax2, emax0, emax1,
				       kmax2, kmax0, kmax1,
				       tmax2, tmax0, tmax1, viewer,true);
		draw_segment_app_ridge(v0, v1, v2,
				       emax0, emax1, emax2,
				       kmax0, kmax1, kmax2,
				       tmax0, tmax1, tmax2, viewer,true);
	}
}

// Draw apparent ridges of the mesh
void draw_mesh_app_ridges(RowVector3d camera_position, const vector<double> &ndotv, const vector<double> &q1,
			  const vector<Vector2d> &t1, const vector<double> &Dt1q1,igl::opengl::glfw::Viewer& viewer)
{
	for(int i = 0; i<F1.rows(); i++)
		draw_face_app_ridges(camera_position, F1(i,0),F1(i,1), F1(i,2),
				     ndotv, q1, t1, Dt1q1, viewer);

}
vector<double> q1,Dt1q1;
		 vector<Vector2d> t1;	
		 vector< double> ndotv;

RowVector3d compute_camera_center(igl::opengl::glfw::Viewer& viewer)
{
		// Compute center and radius of the sphere
		Eigen::RowVector3d center = V1.colwise().mean();
		double radius = 0.;
		for (int i = 0; i < V1.rows(); i++) {
			double dist = (V1.row(i) - center).norm();
			if (dist > radius) {
				radius = dist;
			}
		}
		radius*= radius_scale;
				// Create sphere mesh
		int n_theta = 20;
		int n_phi = 20;
		Eigen::MatrixXd SV(n_theta * (n_phi+1), 3);
		Eigen::MatrixXi SF(2 * n_theta * (n_phi ), 3);

		for (int i = 0; i < n_phi+1; i++) {
			double phi = 2.0 * M_PI * i / n_phi;
			for (int j = 0; j < n_theta; j++) {
			double theta = M_PI * j / (n_theta - 1);
			int index = i * n_theta + j;
			SV.row(index) << radius * sin(theta) * cos(phi),
								radius * sin(theta) * sin(phi),
								radius * cos(theta);
			SV.row(index) += center;
			}
		}


		for (int i = 0; i < n_phi ; i++) {
			for (int j = 0; j < n_theta; j++) {
				int a = i * n_theta + j;
				int b = i * n_theta + (j + 1) % n_theta;
				int c = (i + 1) * n_theta + (j + 1) % n_theta;
				int d = (i + 1) * n_theta + j;

				int fi = 2 * (i * n_theta + j);
				SF.row(fi) << a, b, c;
				SF.row(fi + 1) << a, c, d;
			}
		}
		// Get the screen width and height
		int width = viewer.core().viewport(2);
		int height = viewer.core().viewport(3);
		// Calculate the screen center coordinates
		double screen_center_x =  static_cast<double>(width) / 2.0;
		double screen_center_y = viewer.core().viewport(3) - static_cast<double>(height) / 2.0;
		int fid;
        Eigen::Vector3d bc;
        RowVector3d camera_center;
        if(igl::unproject_onto_mesh(Eigen::Vector2f(screen_center_x,screen_center_y),viewer.core().view,
                                    viewer.core().proj,
                                    viewer.core().viewport,
                                    SV,SF,fid,bc))
            camera_center = SV.row(SF(fid,0))*bc(0) + SV.row(SF(fid,1))*bc(1) + SV.row(SF(fid,2))*bc(2);

		// viewer.data().clear();
		// viewer.data().set_mesh(SV, SF); 
		viewer.data().add_points(camera_center,RowVector3d(1.,0.,0.));
		return camera_center;
}


bool key_down(igl::opengl::glfw::Viewer& viewer, unsigned char key, int modifier)
{
    switch(key)
	{ 
		 
		case '0':
        {
			q1.clear(),t1.clear(),ndotv.clear(),Dt1q1.clear();
			q1.resize(V1.rows(),0.0);
			Dt1q1.resize(V1.rows(),0.0);
			t1.resize(V1.rows(), Vector2d(0.0,0.0));	
			ndotv.resize(V1.rows(), 0.0);
			// std::vector< double> kr(V1.rows(), 0.0);
			viewer.data(0).clear();
			RowVector3d camera_eyep = compute_camera_center(viewer);
		#pragma omp parallel for
            for (int i = 0; i < V1.rows(); i++) 
			{	
				Eigen::RowVector3d viewdir = camera_eyep -RowVector3d(V1.row(i))  ;
				viewdir.normalized();
				ndotv[i] = viewdir.dot(N.row(i));
				// if(fabs(ndotv[i])<=1e-6)
				// ndotv[i] =1e-6;
				double u = viewdir.dot(PD1.row(i)), u2 = u*u;
				double v = viewdir.dot(PD2.row(i)), v2 = v*v;
				double csc2theta = 1.0 / (u2 + v2);
				compute_viewdependent_curvature(V1, i, ndotv[i],u2*csc2theta, u*v*csc2theta, v2*csc2theta,q1[i], t1[i]);
				compute_Dt1q1(V1, i, ndotv[i], q1, t1, Dt1q1[i]);
			}
			draw_mesh_app_ridges(camera_eyep,ndotv, q1, t1, Dt1q1, viewer);
			cout<<"done"<<endl;
        }
		return true;
		case '1':
		{
			if(threshold<0.11)
			threshold+=threshold/2.;
			else if(threshold>2.)
			threshold+=1;
			else
			threshold+=0.1;
			q1.clear(),t1.clear(),ndotv.clear(),Dt1q1.clear();
			q1.resize(V1.rows(),0.0);
			Dt1q1.resize(V1.rows(),0.0);
			t1.resize(V1.rows(), Vector2d(0.0,0.0));	
			ndotv.resize(V1.rows(), 0.0);
			// std::vector< double> kr(V1.rows(), 0.0);
			viewer.data(0).clear();
			RowVector3d camera_eyep = compute_camera_center(viewer);
		#pragma omp parallel for
            for (int i = 0; i < V1.rows(); i++) 
			{
				Eigen::RowVector3d viewdir = RowVector3d(V1.row(i)) - camera_eyep ;
				viewdir.normalized();
				ndotv[i] = viewdir.dot(N.row(i));
				// if(fabs(ndotv[i])<=1e-6)
				// ndotv[i] =1e-6;
				double u = viewdir.dot(PD1.row(i)), u2 = u*u;
				double v = viewdir.dot(PD2.row(i)), v2 = v*v;
				
				// kr[i] =  PV1.row(i) * u2 + PV2.row(i) * v2;
				double csc2theta = 1.0 / (u2 + v2);
				compute_viewdependent_curvature(V1, i, ndotv[i],u2*csc2theta, u*v*csc2theta, v2*csc2theta,q1[i], t1[i]);
				compute_Dt1q1(V1, i, ndotv[i], q1, t1, Dt1q1[i]);
			}
			draw_mesh_app_ridges(camera_eyep, ndotv, q1, t1, Dt1q1, viewer);
			cout<<"thresold: "<<threshold<<endl;
		}
		return true;
		case '2':
		{
			if(threshold<0.11)
			threshold-=threshold/2.;
			else if(threshold>3.)
			threshold-=1;
			else
			threshold-=0.1;
			q1.clear(),t1.clear(),ndotv.clear(),Dt1q1.clear();
			q1.resize(V1.rows(),0.0);
			Dt1q1.resize(V1.rows(),0.0);
			t1.resize(V1.rows(), Vector2d(0.0,0.0));	
			ndotv.resize(V1.rows(), 0.0);
			// std::vector< double> kr(V1.rows(), 0.0);
			viewer.data(0).clear();
			RowVector3d camera_eyep = compute_camera_center(viewer);
		#pragma omp parallel for
            for (int i = 0; i < V1.rows(); i++) 
			{
				Eigen::RowVector3d viewdir = RowVector3d(V1.row(i)) - camera_eyep ;
				viewdir.normalized();
				ndotv[i] = viewdir.dot(N.row(i));
				// if(fabs(ndotv[i])<=1e-6)
				// ndotv[i] =1e-6;
				double u = viewdir.dot(PD1.row(i)), u2 = u*u;
				double v = viewdir.dot(PD2.row(i)), v2 = v*v;
				
				double csc2theta = 1.0 / (u2 + v2);
				compute_viewdependent_curvature(V1, i, ndotv[i],u2*csc2theta, u*v*csc2theta, v2*csc2theta,q1[i], t1[i]);
				compute_Dt1q1(V1, i, ndotv[i], q1, t1, Dt1q1[i]);
			}
			draw_mesh_app_ridges(camera_eyep,ndotv, q1, t1, Dt1q1, viewer);
			cout<<"thresold: "<<threshold<<endl;
		}
		return true;
		
		case '3':
		{
		// Compute center and radius of the sphere
		Eigen::RowVector3d center = V1.colwise().mean();
		double radius = 0.;
		for (int i = 0; i < V1.rows(); i++) {
			double dist = (V1.row(i) - center).norm();
			if (dist > radius) {
				radius = dist;
			}
		}
		radius*=  radius_scale;
				// Create sphere mesh
		int n_theta = 20;
		int n_phi = 20;
		Eigen::MatrixXd SV(n_theta * (n_phi+1), 3);
		Eigen::MatrixXi SF(2 * n_theta * (n_phi ), 3);

		for (int i = 0; i < n_phi+1; i++) {
			double phi = 2.0 * M_PI * i / n_phi;
			for (int j = 0; j < n_theta; j++) {
			double theta = M_PI * j / (n_theta - 1);
			int index = i * n_theta + j;
			SV.row(index) << radius * sin(theta) * cos(phi),
								radius * sin(theta) * sin(phi),
								radius * cos(theta);
			SV.row(index) += center;
			}
		}


		for (int i = 0; i < n_phi ; i++) {
			for (int j = 0; j < n_theta; j++) {
				int a = i * n_theta + j;
				int b = i * n_theta + (j + 1) % n_theta;
				int c = (i + 1) * n_theta + (j + 1) % n_theta;
				int d = (i + 1) * n_theta + j;

				int fi = 2 * (i * n_theta + j);
				SF.row(fi) << a, b, c;
				SF.row(fi + 1) << a, c, d;
			}
		}
		// Get the screen width and height
		int width = viewer.core().viewport(2);
		int height = viewer.core().viewport(3);
		// Calculate the screen center coordinates
		double screen_center_x = static_cast<double>(width) / 2.0;
		double screen_center_y = static_cast<double>(height) / 2.0;
		int fid;
        Eigen::Vector3d bc;
        RowVector3d camera_center;
        if(igl::unproject_onto_mesh(Eigen::Vector2f(screen_center_x,screen_center_y),viewer.core().view,
                                    viewer.core().proj,
                                    viewer.core().viewport,
                                    SV,SF,fid,bc))
            camera_center = SV.row(SF(fid,0))*bc(0) + SV.row(SF(fid,1))*bc(1) + SV.row(SF(fid,2))*bc(2);

		viewer.data().clear();
		viewer.data().set_mesh(V1, F1); 
		// viewer.data().add_points(camera_center,RowVector3d(1.,0.,0.));
		
		}
		return true;
		case '4':
		{
			// Compute the mean curvature (average of the principal curvatures)
			MatrixXd K;
			igl::gaussian_curvature(V1,F1,K); 

			// Eigen::VectorXd H = ;//0.5 * (PV1 + PV2);
			

			// Map the max principle curvature values to colors using the jet colormap
			Eigen::MatrixXd C;
			igl::jet(K, true, C);

			// Set the colors of the mesh
			viewer.data(0).set_mesh(V1, F1);
			viewer.data(0).set_colors(C);

		}
		return true;
		case '5':
		{
			// Compute the mean curvature (average of the principal curvatures)
			Eigen::VectorXd H = 0.5 * (PV1 + PV2);
			

			// Map the mean curvature values to colors using the jet colormap
			Eigen::MatrixXd C;
			igl::jet(H, true, C);

			// Set the colors of the mesh
			viewer.data(0).set_mesh(V1, F1);
			viewer.data(0).set_colors(C);

		}
		return true;
		case '6':
		{
			radius_scale+=0.1;
			cout<<"radius increased ZOOM in/out accordingly and press 0 to run"<< endl;
		}
		return true;
		case '7':
		{
			radius_scale-=0.1;
			cout<<"radius decreased ZOOM in/out accordingly and press 0 to run"<< endl;
		}
		return true;
		/* //To draw normals
		case '8':
		{
			for (int i = 0; i < V1.rows(); i++) {
  				  viewer.data().add_edges(V1.row(i),V1.row(i) + 0.1 * N.row(i),Eigen::RowVector3d(1.0, 0.0, 0.0));
			}
		}
		return true;*/
		/*case '9':// To project on to a plane 
		{
			MatrixXd projpoints1(proj1.size(),3);
			MatrixXd projpoints2(proj2.size(),2);
			int j=0;
			for(int i= 0 ;i<projpoints1.rows()-1;i++)
			{
				projpoints1.row(i) = proj1.at(j);
				projpoints1.row(i+1) = proj2.at(j);
				// projpoints2.row(j) = Vector2i(i,i+1);
				j++;
			}
			MatrixXi con(projpoints1.rows()-1,3);
			for (int i=0; i<con.rows(); i++) {
        con.row(i) = (Vector3i(i,i+1,i));
   			 }
			MatrixXd Pr1,Pr2;
			igl::project(projpoints1,viewer.core().view,viewer.core().proj,viewer.core().viewport,Pr1);
			// igl::project(projpoints2,viewer.core().view,viewer.core().proj,viewer.core().viewport,Pr2);

			viewer.data().clear();
			viewer.data().set_mesh(Pr1,con);//Pr1,projpoints2,RowVector3d(0.,0.,0.));
		}
		return true;*/
		default: break;
	}
	return false;
}



int main(int argc, char *argv[])
{

string file = "../data/bunny.obj";
    if (argc>=2) {
      string w = argv[1];
      file = "../data/" + w + ".obj";
    }
    if (argc>=3) {
      threshold = atoi(argv[2]);
    }
    if (argc>=4) {
      radius_scale = atoi(argv[3]);
    }
   
igl::readOBJ(file,V1,F1);
// igl::readOFF("../data/bunny.off",V1,F1);

igl::principal_curvature(V1, F1, PD1, PD2, PV1, PV2,2);
PD1.normalized();
PD2.normalized();
igl::per_vertex_normals(V1, F1,igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_UNIFORM,N);
// Compute triangle normals
// igl::per_face_normals(V1,F1,triangle_normals);

// N = vertex_normals;
// cout<<N<<"\n\n"<<vertex_normals<<endl;0
igl::vertex_triangle_adjacency(V1.rows(), F1, VF, VFi);
// threshold = igl::avg_edge_length(V1,F1);
if(threshold<0.2) threshold =0.2;
  // Plot the mesh
  igl::opengl::glfw::Viewer viewer;
  viewer.data(0).set_mesh(V1, F1);
//   threshold=60.;
  viewer.data().set_face_based(true);
// for (int i = 0; i < V1.rows(); i++) {
//     viewer.data().add_edges(
//         V1.row(i),
//         V1.row(i) + 0.1 * vertex_normals.row(i),
//         Eigen::RowVector3d(0.0, 0.0, 1.0));  // Set color to red
// }
  viewer.callback_key_down = &key_down;
  viewer.core().background_color = Eigen::Vector4f(1.f, 1.f, 1.f, 1.f);
  viewer.core().camera_zoom = 1/1.20;
  viewer.launch();
}
