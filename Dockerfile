FROM arm64v8/ros:humble

ENV DEBIAN_FRONTEND=noninteractive
ENV DISPLAY=:99

RUN apt-get update && apt-get install -y \
    git \
    curl \
    net-tools \
    socat \
    python3-pip \
    python3-colcon-common-extensions \
    ros-humble-moveit \
    ros-humble-realtime-tools \
    ros-humble-generate-parameter-library \
    ros-humble-py-binding-tools \
    ros-humble-joint-state-publisher-gui \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    liblapack-dev \
    libblas-dev \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get remove -y ros-humble-moveit-servo-* 2>/dev/null || true \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install roarm-sdk==0.1.0

WORKDIR /home/ws
RUN git clone https://github.com/DUDULRX/roarm_ws.git

SHELL ["/bin/bash", "-c"]
RUN source /opt/ros/humble/setup.bash && \
    cd /home/ws/roarm_ws && \
    colcon build --packages-select roarm_msgs && \
    source install/setup.bash && \
    colcon build --packages-select moveit_servo && \
    source install/setup.bash && \
    colcon build --packages-select rviz_marker_tools && \
    source install/setup.bash && \
    colcon build --packages-select moveit_task_constructor_msgs && \
    source install/setup.bash && \
    MAKEFLAGS="-j2" colcon build --packages-select moveit_task_constructor_core && \
    source install/setup.bash && \
    colcon build --packages-select moveit_task_constructor_capabilities && \
    source install/setup.bash && \
    colcon build --packages-select moveit_task_constructor_visualization && \
    source install/setup.bash && \
    colcon build --packages-select roarm_moveit_cmd && \
    source install/setup.bash && \
    colcon build --packages-select roarm_moveit_ikfast_plugins && \
    source install/setup.bash && \
    colcon build --packages-select roarm_moveit_mtc_demo && \
    source install/setup.bash && \
    colcon build --packages-select roarm_moveit_servo && \
    source install/setup.bash && \
    colcon build --packages-select roarm_description roarm_driver roarm_moveit --symlink-install

RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /home/ws/roarm_ws/install/setup.bash" >> ~/.bashrc

ENV ROARM_MODEL=roarm_m2

WORKDIR /home/ws/roarm_ws
CMD ["/bin/bash"]
