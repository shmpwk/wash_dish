#!/usr/bin/env roseus

(ros::load-ros-package "jsk_2019_10_spatula")
(load "package://pr2eus/pr2-interface.l")
(ros::roseus "pr2_send_joints")
(ros::advertise "scrape_left_jacobian" jsk_2019_10_spatula::Jacobian 1)
(ros::advertise "scrape_right_jacobian" jsk_2019_10_spatula::Jacobian 1)
(ros::advertise "semantic_annotation" std_msgs::string 1)
(setq *ri* (instance pr2-interface :init))
(pr2-init t)

(defun publish-semantic-annotation (annotation-str)
  (setq msg (instance std_msgs::string :init))
  (send msg :data annotation-str)
  (ros::ros-info "msg [~A]" (send msg :data))
  (ros::publish "semantic_annotation" msg)
  )


(defun prepare-robot ()
  ;;start position
  (publish-semantic-annotation (format nil "av-new-bowl_start"))
  (send *ri* :angle-vector (send *pr2* :angle-vector) 1000 :larm-controller)
 
  ;;hand the robot the bowl
  (unix:sleep 2)
  (send *pr2* :start-grasp :larm)
  (send *ri* :start-grasp :larm :gain 0.05)
  (send *ri* :wait-interpolation)
  )

(defun get-dense-ptcloud()
  (ros::service-call "passthrough_points/request" (instance std_srvs::EmptyRequest :init))
  (send *pr2* :angle-vector *look-front*)
  (send *ri* :angle-vector (send *pr2* :angle-vector) 1000)
  (send *ri* :wait-interpolation)
  (unix:usleep 10000) ;;sleep 1/100 of a second to make sure /tf is fresh
  ;;passthrough the pointcloud data
  (ros::service-call "passthrough_points/request" (instance std_srvs::EmptyRequest :init))
  (publish-semantic-annotation (format nil "vision_0"))
  
  (send *pr2* :angle-vector *look-right*)
  (send *ri* :angle-vector (send *pr2* :angle-vector) 1000)
  (send *ri* :wait-interpolation)
  (unix:usleep 10000) ;;sleep 1/100 of a second to make sure /tf is fresh
  (ros::service-call "passthrough_points/request" (instance std_srvs::EmptyRequest :init))
  (publish-semantic-annotation (format nil "vision_1"))

  (send *pr2* :angle-vector *look-back*)
  (send *ri* :angle-vector (send *pr2* :angle-vector) 1000)
  (send *ri* :wait-interpolation)
  ;;(unix:sleep 1)
  (ros::service-call "passthrough_points/request" (instance std_srvs::EmptyRequest :init))
  (publish-semantic-annotation (format nil "vision_2"))

  (send *pr2* :angle-vector *look-left*)
  (send *ri* :angle-vector (send *pr2* :angle-vector) 1000)
  (send *ri* :wait-interpolation)
  (unix:usleep 10000) ;;sleep 1/100 of a second to make sure /tf is fresh
  (ros::service-call "passthrough_points/request" (instance std_srvs::EmptyRequest :init))
  (publish-semantic-annotation (format nil "vision_3"))

  (send *pr2* :angle-vector *av-snapshot*)
  (send *ri* :angle-vector (send *pr2* :angle-vector) 1000)
  (send *ri* :wait-interpolation)
  (unix:usleep 10000) ;;sleep 1/100 of a second to make sure /tf is fresh
  (ros::service-call "passthrough_points/request" (instance std_srvs::EmptyRequest :init))
  (publish-semantic-annotation (format nil "vision_end"))
  )

(prepare-robot)
;;(get-dense-ptcloud)
