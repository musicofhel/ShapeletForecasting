"""
Deployment Architecture
Shows how the system is deployed and scaled
"""

from diagrams import Diagram, Cluster, Edge
from diagrams.aws.compute import EC2, ECS, Lambda
from diagrams.aws.database import RDS, ElastiCache
from diagrams.aws.storage import S3
from diagrams.aws.network import ELB, CloudFront, Route53
from diagrams.aws.management import Cloudwatch
from diagrams.aws.security import WAF
from diagrams.onprem.container import Docker
from diagrams.onprem.ci import Jenkins
from diagrams.onprem.vcs import Github

with Diagram("Financial Wavelet Prediction - Deployment Architecture", 
             filename="architecture_diagrams/6_deployment_architecture", 
             show=False,
             direction="TB"):
    
    # DNS and CDN
    dns = Route53("DNS")
    cdn = CloudFront("CDN")
    waf = WAF("Web Application\nFirewall")
    
    # Load Balancer
    alb = ELB("Application\nLoad Balancer")
    
    # Application Tier
    with Cluster("Application Tier"):
        with Cluster("Dashboard Containers"):
            dashboard_1 = ECS("Dashboard\nInstance 1")
            dashboard_2 = ECS("Dashboard\nInstance 2")
            dashboard_3 = ECS("Dashboard\nInstance 3")
            
        with Cluster("API Containers"):
            api_1 = ECS("API\nInstance 1")
            api_2 = ECS("API\nInstance 2")
            api_3 = ECS("API\nInstance 3")
    
    # Processing Tier
    with Cluster("Processing Tier"):
        with Cluster("Batch Processing"):
            batch_processor = EC2("Batch\nProcessor")
            pattern_extractor = EC2("Pattern\nExtractor")
            model_trainer = EC2("Model\nTrainer")
            
        with Cluster("Real-time Processing"):
            realtime_1 = Lambda("Real-time\nProcessor 1")
            realtime_2 = Lambda("Real-time\nProcessor 2")
            
    # Data Tier
    with Cluster("Data Tier"):
        with Cluster("Databases"):
            market_db = RDS("Market Data\n(PostgreSQL)")
            pattern_db = RDS("Pattern Data\n(PostgreSQL)")
            
        with Cluster("Cache"):
            redis_cache = ElastiCache("Redis Cache")
            
        with Cluster("Object Storage"):
            model_storage = S3("Model Storage")
            pattern_storage = S3("Pattern Storage")
            backup_storage = S3("Backup Storage")
    
    # Monitoring and CI/CD
    with Cluster("DevOps"):
        monitoring = Cloudwatch("CloudWatch\nMonitoring")
        ci_cd = Jenkins("Jenkins\nCI/CD")
        github = Github("GitHub\nRepository")
        
    # Container Registry
    with Cluster("Container Management"):
        docker_registry = Docker("Docker\nRegistry")
    
    # Flow - User Access
    dns >> cdn >> waf >> alb
    
    # Flow - Load Distribution
    alb >> Edge(label="Dashboard") >> [dashboard_1, dashboard_2, dashboard_3]
    alb >> Edge(label="API") >> [api_1, api_2, api_3]
    
    # Flow - Processing
    [api_1, api_2, api_3] >> [batch_processor, realtime_1, realtime_2]
    batch_processor >> [pattern_extractor, model_trainer]
    
    # Flow - Data Access
    [dashboard_1, dashboard_2, dashboard_3] >> redis_cache
    redis_cache >> [market_db, pattern_db]
    
    [api_1, api_2, api_3] >> [market_db, pattern_db]
    [pattern_extractor, model_trainer] >> [pattern_storage, model_storage]
    
    # Flow - Real-time
    [realtime_1, realtime_2] >> redis_cache
    
    # Flow - Backup
    [market_db, pattern_db] >> Edge(label="Backup") >> backup_storage
    
    # Flow - Monitoring
    [dashboard_1, api_1, batch_processor, realtime_1, market_db, redis_cache] >> monitoring
    
    # Flow - CI/CD
    github >> ci_cd >> docker_registry
    docker_registry >> [dashboard_1, api_1, batch_processor]
