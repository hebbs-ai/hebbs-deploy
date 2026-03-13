use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use tonic::{Request, Response, Status};

use hebbs_core::engine::Engine;
use hebbs_proto::generated as pb;
use pb::health_check_response::ServingStatus;
use pb::health_service_server::HealthService;

pub struct HealthServiceImpl {
    pub engine: Arc<Engine>,
    pub start_time: Instant,
    pub version: String,
    pub data_dir: PathBuf,
}

#[tonic::async_trait]
impl HealthService for HealthServiceImpl {
    async fn check(
        &self,
        _request: Request<pb::HealthCheckRequest>,
    ) -> Result<Response<pb::HealthCheckResponse>, Status> {
        if !self.data_dir.exists() {
            return Ok(Response::new(pb::HealthCheckResponse {
                status: ServingStatus::NotServing as i32,
                version: self.version.clone(),
                memory_count: 0,
                uptime_seconds: self.start_time.elapsed().as_secs(),
            }));
        }

        let engine = self.engine.clone();
        let count = tokio::task::spawn_blocking(move || engine.count())
            .await
            .map_err(|e| Status::internal(format!("task join error: {}", e)))?
            .unwrap_or(0);

        let uptime = self.start_time.elapsed().as_secs();

        Ok(Response::new(pb::HealthCheckResponse {
            status: ServingStatus::Serving as i32,
            version: self.version.clone(),
            memory_count: count as u64,
            uptime_seconds: uptime,
        }))
    }
}
