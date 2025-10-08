# Modular Badminton Business Running System

A comprehensive, microservices-based system for managing all aspects of a badminton business, from court reservations to equipment maintenance, with data-driven insights and real-time capabilities.

## 🏗️ Architecture Overview

The system follows a microservices architecture with the following components:

### Core Services
- **User Management Service** (Port 8000) - Authentication, authorization, user profiles
- **Court Reservation Service** (Port 8001) - Booking logic, dynamic pricing, waitlist
- **Match Scoring Service** (Port 8002) - Real-time scoring, player stats, match history
- **Inventory & Maintenance Service** (Port 8003) - Product management, equipment tracking
- **Analytics Service** (Port 8004) - Reports, visualizations, data export
- **API Gateway** (Port 8080) - Single entry point, routing, authentication

### Supporting Infrastructure
- **PostgreSQL** - Primary database for all services
- **Redis** - Caching and session storage
- **Docker** - Containerization and orchestration

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.11+ (for local development)
- PostgreSQL (if running locally)

### Using Docker (Recommended)

1. **Clone and navigate to the project:**
   ```bash
   cd badminton_system
   ```

2. **Start all services:**
   ```bash
   docker compose -f docker/docker compose.yml up -d
   ```

3. **Check service health:**
   ```bash
   curl http://localhost:8080/health/services
   ```

4. **Access the API Gateway:**
   - Main API: http://localhost:8080
   - API Documentation: http://localhost:8080/docs

### Local Development

1. **Set up each service individually:**
   ```bash
   # User Service
   cd user_service
   pip install -r requirements.txt
   uvicorn main:app --port 8000 --reload

   # Court Reservation Service
   cd ../court_reservation
   pip install -r requirements.txt
   uvicorn main:app --port 8001 --reload

   # Continue for other services...
   ```

2. **Start the API Gateway:**
   ```bash
   cd api_gateway
   pip install -r requirements.txt
   uvicorn main:app --port 8080 --reload
   ```

## 📋 Features

### ✅ Implemented Features

#### User Management
- [x] User registration and authentication (JWT)
- [x] Role-based access control (admin, user, referee, technician, shopkeeper)
- [x] User profile management
- [x] Secure password hashing

#### Court Reservation
- [x] Court booking with availability checking
- [x] Dynamic pricing (peak/off-peak hours)
- [x] Waitlist management
- [x] Reservation cancellation
- [x] Multiple court types (standard, premium, VIP)

#### Match Scoring
- [x] Player profile creation
- [x] Match setup (singles/doubles)
- [x] Real-time score tracking
- [x] Undo/redo functionality
- [x] BWF rule validation
- [x] WebSocket support for live updates
- [x] Player statistics aggregation
- [x] Match history export (JSON/CSV)

#### Inventory & Maintenance
- [x] Product catalog management
- [x] Stock level monitoring
- [x] Reorder alerts
- [x] Equipment condition tracking
- [x] Maintenance scheduling
- [x] Predictive maintenance using wear models
- [x] Technician assignment

#### Analytics & Reporting
- [x] Court utilization reports
- [x] Player performance analytics
- [x] Inventory turnover analysis
- [x] Interactive dashboards
- [x] Data visualization (charts, heatmaps)
- [x] Data export capabilities

#### System Infrastructure
- [x] API Gateway with service routing
- [x] Health monitoring
- [x] Docker containerization
- [x] Comprehensive logging
- [x] Error handling
- [x] CORS support

## 🔧 API Endpoints

### Authentication
- `POST /auth/register` - Register new user
- `POST /auth/login` - User login
- `GET /auth/me` - Get current user profile

### Court Management
- `GET /courts` - List all courts
- `POST /reservations` - Create reservation
- `GET /reservations/{id}` - Get reservation details
- `PUT /reservations/{id}/cancel` - Cancel reservation
- `GET /availability` - Check court availability

### Match Management
- `POST /players` - Create player profile
- `GET /players` - List all players
- `POST /matches` - Create new match
- `POST /matches/{id}/start` - Start match
- `POST /matches/{id}/score` - Update score
- `POST /matches/{id}/undo` - Undo last score

### Inventory
- `GET /products` - List products
- `POST /products` - Add new product
- `GET /equipment` - List equipment
- `POST /maintenance` - Schedule maintenance
- `GET /reports/reorder-alerts` - Get reorder alerts

### Analytics
- `POST /reports/court-utilization` - Court utilization report
- `POST /reports/player-performance` - Player performance report
- `GET /dashboard/court-heatmap` - Court usage heatmap
- `GET /export/data` - Export data (JSON/CSV)

### System
- `GET /health` - Gateway health check
- `GET /health/services` - All services health
- `GET /system/info` - System information

## 🏃‍♂️ Usage Examples

### 1. Register a User
```bash
curl -X POST "http://localhost:8080/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "john_doe",
    "email": "john@example.com",
    "password": "securepass123",
    "role": "user"
  }'
```

### 2. Login and Get Token
```bash
curl -X POST "http://localhost:8080/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=john_doe&password=securepass123"
```

### 3. Make a Court Reservation
```bash
curl -X POST "http://localhost:8080/reservations" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "court_id": 1,
    "start_time": "2024-01-20T18:00:00",
    "end_time": "2024-01-20T20:00:00",
    "player_count": 2,
    "skill_level": "intermediate"
  }'
```

### 4. Create a Match
```bash
curl -X POST "http://localhost:8080/matches" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "match_type": "singles",
    "player_ids": [1, 2],
    "live_scoreboard": true
  }'
```

### 5. Generate Court Utilization Report
```bash
curl -X POST "http://localhost:8080/reports/court-utilization?start_date=2024-01-01T00:00:00&end_date=2024-01-31T23:59:59&include_chart=true" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

## 🔒 Security Features

- **JWT Authentication** - Secure token-based authentication
- **Role-Based Access Control** - Different permissions for different user types
- **Password Hashing** - Bcrypt for secure password storage
- **Input Validation** - Pydantic models for request validation
- **CORS Protection** - Configurable cross-origin resource sharing
- **Rate Limiting** - Built into API Gateway (configurable)

## 📊 Monitoring & Observability

- **Health Checks** - All services expose health endpoints
- **Structured Logging** - Comprehensive logging across all services
- **Service Discovery** - API Gateway monitors service availability
- **Error Tracking** - Centralized error handling and reporting

## 🧪 Testing

### Manual Testing
Each service exposes interactive API documentation at `/docs`:
- User Service: http://localhost:8000/docs
- Court Service: http://localhost:8001/docs
- Match Service: http://localhost:8002/docs
- Inventory Service: http://localhost:8003/docs
- Analytics Service: http://localhost:8004/docs
- API Gateway: http://localhost:8080/docs

### Sample Data
Services automatically create sample data on startup for testing purposes.

## 🔧 Configuration

### Environment Variables
Each service can be configured using environment variables:

```bash
# Database
DATABASE_URL=postgresql+psycopg2://user:pass@host:port/db

# JWT
SECRET_KEY=your-secret-key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Logging
LOG_LEVEL=INFO

# Service URLs (API Gateway)
USER_SERVICE_URL=http://user_service:8000
COURT_SERVICE_URL=http://court_service:8001
# ... etc
```

## 🚀 Deployment

### Production Deployment
1. **Update environment variables** in `docker/docker compose.yml`
2. **Configure secrets management** (replace hardcoded secrets)
3. **Set up reverse proxy** (nginx) for SSL termination
4. **Configure monitoring** (Prometheus, Grafana)
5. **Set up log aggregation** (ELK stack)

### Scaling
- Each service can be scaled independently
- Use Kubernetes for advanced orchestration
- Implement load balancing for high availability

## 📈 Performance Considerations

- **Database Indexing** - Proper indexes on frequently queried columns
- **Connection Pooling** - SQLAlchemy connection pools
- **Caching** - Redis for session and frequently accessed data
- **Async Operations** - FastAPI's async capabilities for I/O operations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Check the API documentation at `/docs` endpoints
- Review the logs for error details
- Ensure all services are healthy via `/health/services`

## 🔮 Future Enhancements

- **Mobile App Integration** - REST API ready for mobile clients
- **Payment Processing** - Integration with payment gateways
- **Tournament Management** - Advanced tournament bracket generation
- **Coach Booking** - Skill-based coach matching system
- **Membership Management** - QR code check-in system
- **Plugin System** - Modular plugin architecture
- **Advanced Analytics** - Machine learning for predictive insights
- **Multi-tenant Support** - Support for multiple badminton facilities

---

**Built with ❤️ for the badminton community**
