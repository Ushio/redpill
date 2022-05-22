#pragma once

#include <memory>
#include <map>
#include <queue>
#include <functional>
#include <stdarg.h>
#include <d3d12.h>
#include <dxgi1_6.h>

#include "d3d12shader.h"
#include "d3dx12.h"
#include "dxcapi.h"

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

#define DX_ASSERT( status, message )                                                             \
	if( ( status ) == 0 )                                                                        \
	{                                                                                            \
		char buffer[512];                                                                        \
		snprintf( buffer, sizeof( buffer ), "%s, %s (%d line)\n", message, __FILE__, __LINE__ ); \
		__debugbreak();                                                                          \
	}

namespace dx
{
template <class T>
class DxPtr
{
public:
	DxPtr() {}
	DxPtr( T* ptr ) : _ptr( ptr )
	{
	}
	DxPtr( const DxPtr<T>& rhs ) : _ptr( rhs._ptr )
	{
		if( _ptr )
		{
			_ptr->AddRef();
		}
	}
	DxPtr<T>& operator=( const DxPtr<T>& rhs )
	{
		auto p = _ptr;

		if( rhs._ptr )
		{
			rhs._ptr->AddRef();
		}
		_ptr = rhs._ptr;

		if( p )
		{
			p->Release();
		}
		return *this;
	}
	~DxPtr()
	{
		if( _ptr )
		{
			_ptr->Release();
		}
	}
	T* get()
	{
		return _ptr;
	}
	const T* get() const
	{
		return _ptr;
	}
	T* operator->()
	{
		return _ptr;
	}
	T** getAddressOf()
	{
		return &_ptr;
	}
	operator bool()
	{
		return _ptr != nullptr;
	}

private:
	T* _ptr = nullptr;
};

inline int div_round_up( int val, int divisor )
{
	return ( val + divisor - 1 ) / divisor;
}
inline int next_multiple( int val, int divisor )
{
	return div_round_up( val, divisor ) * divisor;
}
inline std::wstring string_to_wstring( const std::string& s )
{
	int in_length = (int)s.length();
	int out_length = MultiByteToWideChar( CP_ACP, 0, s.c_str(), in_length, 0, 0 );
	std::vector<wchar_t> buffer( out_length );
	if( out_length )
	{
		MultiByteToWideChar( CP_ACP, 0, s.c_str(), in_length, &buffer[0], out_length );
	}
	return std::wstring( buffer.begin(), buffer.end() );
}
// [0] is typically the primary display adapter
inline std::vector<DxPtr<IDXGIAdapter>> allAdapters()
{
	HRESULT hr;

	DxPtr<IDXGIFactory7> factory;
	UINT flagsDXGI = DXGI_CREATE_FACTORY_DEBUG;
	hr = CreateDXGIFactory2( flagsDXGI, IID_PPV_ARGS( factory.getAddressOf() ) );
	DX_ASSERT( hr == S_OK, "" );

	std::vector<DxPtr<IDXGIAdapter>> adapters;

	int adapterIndex = 0;
	for( ;; )
	{
		DxPtr<IDXGIAdapter> adapter;
		hr = factory->EnumAdapters( adapterIndex++, adapter.getAddressOf() );
		if( hr == S_OK )
		{
			adapters.push_back( adapter );
			continue;
		}
		DX_ASSERT( hr == DXGI_ERROR_NOT_FOUND, "" );
		break;
	};

	return adapters;
}
inline void activateDebugLayer()
{
	DxPtr<ID3D12Debug> debugController;
	if( D3D12GetDebugInterface( IID_PPV_ARGS( debugController.getAddressOf() ) ) != S_OK )
	{
		return;
	}
	debugController->EnableDebugLayer();

	DxPtr<ID3D12Debug3> debug;
	if( debugController->QueryInterface( IID_PPV_ARGS( debug.getAddressOf() ) ) != S_OK )
	{
		return;
	}
	debug->SetEnableGPUBasedValidation( true );
}

class Fence
{
public:
	Fence( const Fence& ) = delete;
	void operator=( const Fence& ) = delete;

	Fence( ID3D12Device* d3d12Device, ID3D12CommandQueue* d3d12Queue )
	{
		HRESULT hr;
		hr = d3d12Device->CreateFence( 0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS( _fence.getAddressOf() ) );
		DX_ASSERT( hr == S_OK, "" );
		hr = d3d12Queue->Signal( _fence.get(), 1 );
		DX_ASSERT( hr == S_OK, "" );
	}
	void wait()
	{
		HANDLE e = CreateEvent( nullptr, false, false, nullptr );
		_fence->SetEventOnCompletion( 1, e );
		WaitForSingleObject( e, INFINITE );
		CloseHandle( e );
	}

private:
	DxPtr<ID3D12Fence> _fence;
};

class CommandList
{
public:
	CommandList( const CommandList& ) = delete;
	void operator=( const CommandList& ) = delete;

	CommandList( ID3D12Device* device )
	{
		HRESULT hr;
		hr = device->CreateCommandAllocator( D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS( m_allocator.getAddressOf() ) );
		DX_ASSERT( hr == S_OK, "" );

		hr = device->CreateCommandList(
			0,
			D3D12_COMMAND_LIST_TYPE_DIRECT,
			m_allocator.get(),
			nullptr, /* pipeline state */
			IID_PPV_ARGS( m_list.getAddressOf() ) );
		DX_ASSERT( hr == S_OK, "" );
	}
	ID3D12GraphicsCommandList* d3d12CommandList()
	{
		return m_list.get();
	}
	void scopedStoreCommand( std::function<void( ID3D12GraphicsCommandList* commandList )> f )
	{
		if( m_isClosed )
		{
			m_list->Reset( m_allocator.get(), nullptr );
		}

		f( m_list.get() );
		m_list->Close();
		m_isClosed = true;
	}

	// be careful
	// https://docs.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12commandallocator-reset
	void clear()
	{
		m_allocator->Reset();
	}

private:
	bool m_isClosed = false;
	DxPtr<ID3D12CommandAllocator> m_allocator;
	DxPtr<ID3D12GraphicsCommandList> m_list;
};

class Buffer;
class Device
{
public:
	Device( const Device& ) = delete;
	void operator=( const Device& ) = delete;

	Device( DxPtr<IDXGIAdapter> adapter ) : m_adapter( adapter )
	{
		HRESULT hr;

		struct DeviceIID
		{
			IID iid;
			const char* type;
		};
#define DEVICE_VER( type )      \
	{                           \
		__uuidof( type ), #type \
	}
		const DeviceIID deviceIIDs[] = {
			DEVICE_VER( ID3D12Device8 ),
			DEVICE_VER( ID3D12Device7 ),
			DEVICE_VER( ID3D12Device6 ),
			DEVICE_VER( ID3D12Device5 ),
			DEVICE_VER( ID3D12Device4 ),
			DEVICE_VER( ID3D12Device3 ),
			DEVICE_VER( ID3D12Device2 ),
			DEVICE_VER( ID3D12Device1 ),
		};
#undef DEVICE_VER

		for( auto deviceIID : deviceIIDs )
		{
			hr = D3D12CreateDevice( adapter.get(), D3D_FEATURE_LEVEL_12_0, deviceIID.iid, (void**)m_device.getAddressOf() );
			if( hr == S_OK )
			{
				break;
			}
		}
		DX_ASSERT( hr == S_OK, "" );

		D3D12_FEATURE_DATA_SHADER_MODEL shaderModelFeature = {};
		shaderModelFeature.HighestShaderModel = D3D_SHADER_MODEL_6_6;
		hr = m_device->CheckFeatureSupport( D3D12_FEATURE_SHADER_MODEL, &shaderModelFeature, sizeof( shaderModelFeature ) );
		DX_ASSERT( hr == S_OK, "" );

		m_ShaderModel = shaderModelFeature.HighestShaderModel;

		D3D12_COMMAND_QUEUE_DESC commandQueueDesk = {};
		commandQueueDesk.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
		commandQueueDesk.Priority = D3D12_COMMAND_QUEUE_PRIORITY_HIGH;
		commandQueueDesk.Flags = D3D12_COMMAND_QUEUE_FLAG_DISABLE_GPU_TIMEOUT;
		commandQueueDesk.NodeMask = 0;
		hr = m_device->CreateCommandQueue( &commandQueueDesk, IID_PPV_ARGS( m_queue.getAddressOf() ) );
		DX_ASSERT( hr == S_OK, "" );

		m_commandListsCurrent = std::unique_ptr<CommandList>( new CommandList( m_device.get() ) );
		m_commandListsSleeping = std::unique_ptr<CommandList>( new CommandList( m_device.get() ) );

		DxPtr<IDXGIFactory4> pDxgiFactory;
		hr = CreateDXGIFactory1( __uuidof( IDXGIFactory1 ), (void**)pDxgiFactory.getAddressOf() );
		DX_ASSERT( hr == S_OK, "" );

		DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {};
		swapChainDesc.BufferCount = 2;
		swapChainDesc.Width = 64;
		swapChainDesc.Height = 64;
		swapChainDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
		swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
		swapChainDesc.SampleDesc.Count = 1;

		// Create the swap chain
		hr = pDxgiFactory->CreateSwapChainForComposition( m_queue.get(), &swapChainDesc, nullptr, m_swapchain.getAddressOf() );
		DX_ASSERT( hr == S_OK, "" );

		hr = m_device->CreateCommittedResource(
			&CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_UPLOAD ),
			D3D12_HEAP_FLAG_NONE,
			&CD3DX12_RESOURCE_DESC::Buffer( IO_CHANK_BYTES ),
			D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr,
			IID_PPV_ARGS( m_fragmentBuffer.getAddressOf() ) );
		DX_ASSERT( hr == S_OK, "" );

		D3D12_RANGE range = {};
		hr = m_fragmentBuffer->Map( 0, &range, &m_fragmentPointer );
		DX_ASSERT( hr == S_OK, "" );
		m_fragmentBuffer->SetName( L"Fragment Buffer" );

		// readback heap
		for( int i = 0; i < 2; i++ )
		{
			hr = m_device->CreateCommittedResource(
				&CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_READBACK ),
				D3D12_HEAP_FLAG_NONE,
				&CD3DX12_RESOURCE_DESC::Buffer( IO_CHANK_BYTES ),
				D3D12_RESOURCE_STATE_COPY_DEST,
				nullptr,
				IID_PPV_ARGS( m_readbackBuffers[i].getAddressOf() ) );
			DX_ASSERT( hr == S_OK, "" );

			D3D12_RANGE range = {};
			hr = m_readbackBuffers[i]->Map( 0, &range, &m_readbackPointers[i] );
			DX_ASSERT( hr == S_OK, "" );
		}
		m_readbackBuffers[0]->SetName( L"m_readbackBuffers[0]" );
		m_readbackBuffers[1]->SetName( L"m_readbackBuffers[1]" );
	}
	~Device()
	{
		wait();
	}
	ID3D12Device* d3d12Device()
	{
		return m_device.get();
	}
	ID3D12CommandQueue* d3d12Queue()
	{
		return m_queue.get();
	}

	// https://docs.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12device-setstablepowerstate
	void setStablePowerState( bool enabled )
	{
		HRESULT hr = m_device->SetStablePowerState( enabled );
		DX_ASSERT( hr == S_OK, "" );
	}

	void present()
	{
		HRESULT hr;
		hr = m_swapchain->Present( 1, 0 );
		DX_ASSERT( hr == S_OK, "" );
	}
	std::wstring name() const
	{
		DXGI_ADAPTER_DESC d;
		HRESULT hr = m_adapter->GetDesc( &d );
		DX_ASSERT( hr == S_OK, "" );
		return d.Description;
	}

	enum class CopyMode
	{
		PrefferedSync,
		PrefferedEnqueue
	};
	void copyH2D( Buffer* buffer, const void* src, int64_t dstOffset, int64_t bytes, CopyMode copyMode = CopyMode::PrefferedSync );
	void copyD2H( void* dst, Buffer* buffer, int64_t srcOffset, int64_t bytes );

	template <class T>
	void copyD2H( std::vector<T>* dst, Buffer* buffer )
	{
		dst->resize( buffer->bytes() / sizeof( T ) );
		copyD2H( dst->data(), buffer, 0, buffer->bytes() );
	}

	void enqueueCommand( std::function<void( ID3D12GraphicsCommandList* commandList )> f )
	{
		if( 32 < m_commandCounter++ )
		{
			m_commandCounter = 0;
			if( m_commandSleepingFence )
			{
				m_commandSleepingFence->wait();
				m_commandListsSleeping->clear();
			}

			std::swap( m_commandListsCurrent, m_commandListsSleeping );

			m_commandSleepingFence = std::unique_ptr<Fence>( newFence() );
		}
		m_commandListsCurrent->scopedStoreCommand( f );

		ID3D12CommandList* const command[] = { m_commandListsCurrent->d3d12CommandList() };
		m_queue->ExecuteCommandLists( 1, command );
	}
	Fence *newFence()
	{
		return new Fence( m_device.get(), m_queue.get() );
	}

	void wait()
	{
		std::unique_ptr<Fence> f( newFence() );
		f->wait();
	}

private:
	mutable DxPtr<IDXGIAdapter> m_adapter;
	D3D_SHADER_MODEL m_ShaderModel;
	DxPtr<ID3D12Device> m_device;
	DxPtr<ID3D12CommandQueue> m_queue;
	DxPtr<IDXGISwapChain1> m_swapchain;

	int m_commandCounter = 0;
	std::unique_ptr<Fence> m_commandSleepingFence;
	std::unique_ptr<CommandList> m_commandListsCurrent;
	std::unique_ptr<CommandList> m_commandListsSleeping;

	enum
	{
		IO_CHANK_BYTES = 1024 * 1024 * 128
	};

	DxPtr<ID3D12Resource> m_fragmentBuffer;
	void* m_fragmentPointer;
	int m_fragmentCopyingHead = 0;
	int m_fragmentCopyingTail = 0;
	struct CopyTask
	{
		int bytes;
		std::shared_ptr<Fence> fence;
	};
	std::queue<CopyTask> m_fragmentCopyQueue;

	DxPtr<ID3D12Resource> m_readbackBuffers[2];
	void* m_readbackPointers[2];
};

class DeviceStopwatch
{
public:
	DeviceStopwatch( Device* device, int capacity, bool enableStopwatch = true ) : m_device( device ), m_capacity( capacity ), m_enableStopwatch( enableStopwatch ), m_readbackPtr( 0 )
	{
		if( m_enableStopwatch == false )
			return;

		HRESULT hr;
		
		D3D12_QUERY_HEAP_DESC desc = {};
		desc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
		desc.Count = capacity;
		hr = device->d3d12Device()->CreateQueryHeap( &desc, IID_PPV_ARGS( m_queryHeap.getAddressOf() ) );
		DX_ASSERT( hr == S_OK, "" );

		hr = device->d3d12Device()->CreateCommittedResource(
			&CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_READBACK ),
			D3D12_HEAP_FLAG_NONE,
			&CD3DX12_RESOURCE_DESC::Buffer( sizeof( uint64_t ) * m_capacity ),
			D3D12_RESOURCE_STATE_COPY_DEST,
			nullptr,
			IID_PPV_ARGS( m_readbackBuffer.getAddressOf() ) );
		DX_ASSERT( hr == S_OK, "" );

		D3D12_RANGE range = {};
		hr = m_readbackBuffer->Map( 0, &range, &m_readbackPtr );
		DX_ASSERT( hr == S_OK, "" );
	}
	void begin( const char* format, ... )
	{
		if( m_enableStopwatch == false )
			return;

		va_list ap;
		va_start( ap, format );
		char text[256];
		int bytes = vsnprintf( text, sizeof( text ), format, ap );
		va_end( ap );
		m_label = text;

		if( m_records.count( m_label ) == 0 )
		{
			m_records[m_label].begin = m_locationIndexer++;
			m_records[m_label].end   = m_locationIndexer++;
		}

		m_device->enqueueCommand(
			[&]( ID3D12GraphicsCommandList* commandList )
			{
				commandList->EndQuery( m_queryHeap.get(), D3D12_QUERY_TYPE_TIMESTAMP, m_records[m_label].begin );
			} );
	}
	void end()
	{
		if( m_enableStopwatch == false )
			return;

		DX_ASSERT( m_records.count( m_label ), "" );

		m_device->enqueueCommand(
			[&]( ID3D12GraphicsCommandList* commandList )
			{
				commandList->EndQuery( m_queryHeap.get(), D3D12_QUERY_TYPE_TIMESTAMP, m_records[m_label].end );
			} );

		m_label = "";
	}
	void collect() 
	{
		if( m_enableStopwatch == false )
			return;

		m_device->enqueueCommand(
			[&]( ID3D12GraphicsCommandList* commandList )
			{
				commandList->ResolveQueryData( m_queryHeap.get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, m_locationIndexer, m_readbackBuffer.get(), 0 );
			} );

		std::unique_ptr<Fence> fence( m_device->newFence() );
		fence->wait();

		uint64_t freq = 1;
		HRESULT hr;
		hr = m_device->d3d12Queue()->GetTimestampFrequency( &freq );
		DX_ASSERT( hr == S_OK, "" );

		const uint64_t* timestamps = (const uint64_t *)m_readbackPtr;

		for( auto it = m_records.begin(); it != m_records.end() ; ++it )
		{
			uint64_t b = timestamps[it->second.begin];
			uint64_t e = timestamps[it->second.end];
			uint64_t deltaTime = e - b;
			double durationS = (double)deltaTime / (double)freq;
			it->second.durationMS = durationS * 1000.0;
		}
	}

	double ms( const char* format, ... ) const 
	{
		if( m_enableStopwatch == false )
			return 0.0;

		va_list ap;
		va_start( ap, format );
		char text[256];
		int bytes = vsnprintf( text, sizeof( text ), format, ap );
		va_end( ap );

		DX_ASSERT( m_records.count( text ), "" );

		auto it = m_records.find( text );
		return it->second.durationMS;
	}
private:
	Device* m_device;
	int m_capacity;
	bool m_enableStopwatch;
	DxPtr<ID3D12QueryHeap> m_queryHeap;
	DxPtr<ID3D12Resource> m_readbackBuffer;
	void* m_readbackPtr;

	int m_locationIndexer = 0;
	std::string m_label;

	struct Record
	{
		int begin = 0;
		int end = 0;
		double durationMS = 0.0;
	};
	std::map<std::string, Record> m_records;
};

/*
	 data is D3D12_HEAP_TYPE_UPLOAD pointer. please make sure this object is alive during shader execution.
	*/
class ConstantBuffer
{
public:
	ConstantBuffer( const ConstantBuffer& ) = delete;
	void operator=( const ConstantBuffer& ) = delete;

	ConstantBuffer( Device* device, int bytes, const wchar_t* name = 0 )
		: _bytes( next_multiple( bytes, 256 ) ), _typeBytes( bytes )
	{
		DX_ASSERT( 1 <= bytes, "T shouldn't be empty" );

		HRESULT hr;
		hr = device->d3d12Device()->CreateCommittedResource(
			&CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_UPLOAD ),
			D3D12_HEAP_FLAG_NONE,
			&CD3DX12_RESOURCE_DESC::Buffer( _bytes ),
			D3D12_RESOURCE_STATE_GENERIC_READ,
			nullptr,
			IID_PPV_ARGS( _resource.getAddressOf() ) );
		DX_ASSERT( hr == S_OK, "" );

		if( name )
		{
			_resource->SetName( name );
		}

		D3D12_RANGE range = {};
		void* p;
		hr = _resource->Map( 0, &range, &p );
		DX_ASSERT( hr == S_OK, "" );
		_ptr = p;
	}
	~ConstantBuffer()
	{
		D3D12_RANGE range = {};
		_resource->Unmap( 0, &range );
	}
	ID3D12Resource* d3d12Resource()
	{
		return _resource.get();
	}
	int64_t bytes() const
	{
		return _bytes;
	}
	int64_t typeBytes() const
	{
		return _typeBytes;
	}
	void* ptr()
	{
		return _ptr;
	}

private:
	void* _ptr;
	int64_t _bytes = 0;
	int64_t _typeBytes = 0;
	DxPtr<ID3D12Resource> _resource;
};

class Buffer
{
public:
	Buffer( const Buffer& ) = delete;
	void operator=( const Buffer& ) = delete;

	Buffer( Device* device, int64_t bytes, const char* format, ... )
		: m_bytes( std::max( bytes, 1LL ) )
	{
		HRESULT hr;
		hr = device->d3d12Device()->CreateCommittedResource(
			&CD3DX12_HEAP_PROPERTIES( D3D12_HEAP_TYPE_DEFAULT ),
			D3D12_HEAP_FLAG_NONE /* I don't know */,
			&CD3DX12_RESOURCE_DESC::Buffer( m_bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS ),
			D3D12_RESOURCE_STATE_COMMON,
			nullptr,
			IID_PPV_ARGS( _resource.getAddressOf() ) );
		DX_ASSERT( hr == S_OK, "" );

		if( format )
		{
			va_list ap;
			va_start( ap, format );
			char text[256];
			int bytes = vsnprintf( text, sizeof( text ), format, ap );
			va_end( ap );
			m_name = text;
			_resource->SetName( string_to_wstring( m_name ).c_str() );
		}
	}
	int64_t bytes() const
	{
		return m_bytes;
	}
	ID3D12Resource* d3d12Resource()
	{
		return _resource.get();
	}

private:
	int64_t m_bytes;
	DxPtr<ID3D12Resource> _resource;
	std::string m_name;
};

void Device::copyH2D( Buffer* buffer, const void* src, int64_t dstOffset, int64_t bytes, CopyMode copyMode )
{
	DX_ASSERT( dstOffset + bytes <= buffer->bytes(), "" );

	int64_t dstPtr = dstOffset;
	int64_t srcPtr = 0;
	int64_t reminder = bytes;
	for( ;; )
	{
		int fragmentTo = ( m_fragmentCopyingHead % IO_CHANK_BYTES );
		int maxBatch = IO_CHANK_BYTES - fragmentTo;
		int64_t batch = std::min<int64_t>( maxBatch, reminder );

		int available = IO_CHANK_BYTES - ( m_fragmentCopyingHead - m_fragmentCopyingTail );
		while( available < batch )
		{
			CopyTask task = m_fragmentCopyQueue.front();
			m_fragmentCopyQueue.pop();
			task.fence->wait();
			m_fragmentCopyingTail += task.bytes;
			available = IO_CHANK_BYTES - ( m_fragmentCopyingHead - m_fragmentCopyingTail );

			if( IO_CHANK_BYTES < m_fragmentCopyingTail && IO_CHANK_BYTES < m_fragmentCopyingHead )
			{
				m_fragmentCopyingTail -= IO_CHANK_BYTES;
				m_fragmentCopyingHead -= IO_CHANK_BYTES;
			}
		}

		memcpy( (char*)m_fragmentPointer + fragmentTo, (const char*)src + srcPtr, batch );
		enqueueCommand(
			[&]( ID3D12GraphicsCommandList* commandList )
			{
				commandList->CopyBufferRegion(
					buffer->d3d12Resource(), dstPtr,
					m_fragmentBuffer.get(), fragmentTo, batch );
			} );

		m_fragmentCopyingHead += batch;

		CopyTask task;
		task.fence = std::shared_ptr<Fence>( newFence() );
		task.bytes = batch;
		m_fragmentCopyQueue.push( task );

		reminder -= batch;
		if( reminder == 0 )
		{
			break;
		}
		dstPtr += batch;
		srcPtr += batch;
	}

	if( copyMode == CopyMode::PrefferedSync )
	{
		while( !m_fragmentCopyQueue.empty() )
		{
			CopyTask task = m_fragmentCopyQueue.front();
			m_fragmentCopyQueue.pop();
			task.fence->wait();
		}
		m_fragmentCopyingTail = 0;
		m_fragmentCopyingHead = 0;
	}
}

void Device::copyD2H( void* dst, Buffer* buffer, int64_t srcOffset, int64_t bytes )
{
	DX_ASSERT( srcOffset + bytes <= buffer->bytes(), "" );

	int64_t dstPtr = 0;
	int64_t srcPtr = srcOffset;
	int64_t reminder = bytes;
	int index = 0;
	std::unique_ptr<Fence> fence;

	void* readbackPtr = 0;
	int64_t readbackBytes = 0;
	int64_t readbackDst = 0;

	for( ;; )
	{
		if( fence )
		{
			fence->wait();
			memcpy( (char*)dst + readbackDst, readbackPtr, readbackBytes );
			if( reminder == 0 )
				break;
		}

		int64_t batch = std::min<int64_t>( IO_CHANK_BYTES, reminder );
		enqueueCommand(
			[&]( ID3D12GraphicsCommandList* commandList )
			{
				commandList->CopyBufferRegion(
					m_readbackBuffers[index % 2].get(), 0,
					buffer->d3d12Resource(), srcPtr, batch );
			} );
		readbackPtr = m_readbackPointers[index % 2];
		readbackBytes = batch;
		readbackDst = dstPtr;

		fence = std::unique_ptr<Fence>( newFence() );

		reminder -= batch;
		dstPtr += batch;
		srcPtr += batch;
		index++;
	}
}

class Compiler
{
public:
	Compiler()
	{
		HRESULT hr;
		hr = DxcCreateInstance( CLSID_DxcUtils, IID_PPV_ARGS( _dxUtils.getAddressOf() ) );
		DX_ASSERT( hr == S_OK, "" );
		hr = DxcCreateInstance( CLSID_DxcCompiler, IID_PPV_ARGS( _dxCompiler.getAddressOf() ) );
		DX_ASSERT( hr == S_OK, "" );
	}
	static Compiler& compiler()
	{
		static Compiler c;
		return c;
	}
	IDxcUtils* dxUtils()
	{
		return _dxUtils.get();
	}
	IDxcCompiler3* dxCompiler()
	{
		return _dxCompiler.get();
	}

private:
	DxPtr<IDxcUtils> _dxUtils;
	DxPtr<IDxcCompiler3> _dxCompiler;
};
enum class CompileMode
{
	Release,
	Debug
};
inline void loadAsVector( std::vector<char>* buffer, const char* fllePath )
{
	FILE* fp = fopen( fllePath, "rb" );
	if( fp == nullptr )
	{
		return;
	}

	fseek( fp, 0, SEEK_END );

	buffer->resize( ftell( fp ) );

	fseek( fp, 0, SEEK_SET );

	size_t s = fread( buffer->data(), 1, buffer->size(), fp );
	if( s != buffer->size() )
	{
		buffer->clear();
		return;
	}
	fclose( fp );
	fp = nullptr;
}
class Shader
{
public:
	Shader( Device* device, const char* filename, const char* includeDir, const std::vector<std::string>& extraArgs, CompileMode compileMode )
	{
		HRESULT hr;
		DxPtr<IDxcIncludeHandler> pIncludeHandler;
		hr = Compiler::compiler().dxUtils()->CreateDefaultIncludeHandler( pIncludeHandler.getAddressOf() );
		DX_ASSERT( hr == S_OK, "" );

		std::wstring NAME = string_to_wstring( std::string( filename ) );
		std::wstring I = string_to_wstring( std::string( includeDir ) );
		std::vector<const wchar_t*> args = {
			NAME.c_str(),
			L"-T",
			L"cs_6_5",
			L"-I",
			I.c_str(),
		};
		std::vector<std::wstring> extras;
		for( const std::string& s : extraArgs )
		{
			extras.push_back( string_to_wstring( s ) );
		}
		for( const std::wstring& s : extras )
		{
			args.push_back( s.c_str() );
		}

		if( compileMode == CompileMode::Debug )
		{
			args.push_back( L"-Zi" );			// Enable debug information
			args.push_back( L"-Od" );			// Disable optimizations
			args.push_back( L"-Qembed_debug" ); // Embed PDB in shader container (must be used with /Zi)
		}

		std::vector<char> src;
		loadAsVector( &src, filename );
		DX_ASSERT( src.size() != 0, "" );

		DxcBuffer buffer = {};
		buffer.Ptr = src.data();
		buffer.Size = src.size();
		buffer.Encoding = DXC_CP_ACP;

		DxPtr<IDxcResult> compileResult;
		hr = Compiler::compiler().dxCompiler()->Compile(
			&buffer,
			args.data(),
			args.size(),
			pIncludeHandler.get(),
			IID_PPV_ARGS( compileResult.getAddressOf() ) // Compiler output status, buffer, and errors.
		);
		DX_ASSERT( hr == S_OK, "" );

		DxPtr<IDxcBlobUtf8> compileErrors;
		hr = compileResult->GetOutput( DXC_OUT_ERRORS, IID_PPV_ARGS( compileErrors.getAddressOf() ), nullptr );
		DX_ASSERT( hr == S_OK, "" );
		if( compileErrors.get() && compileErrors->GetStringLength() != 0 )
		{
			printf( "Warnings and Errors:\n%s\n", compileErrors->GetStringPointer() );
		}

		DxPtr<IDxcBlob> objectBlob;
		hr = compileResult->GetOutput( DXC_OUT_OBJECT, IID_PPV_ARGS( objectBlob.getAddressOf() ), nullptr );
		DX_ASSERT( hr == S_OK, "" );

		DxPtr<IDxcContainerReflection> reflectionContainer;
		UINT32 shaderIdx;
		hr = DxcCreateInstance( CLSID_DxcContainerReflection, IID_PPV_ARGS( reflectionContainer.getAddressOf() ) );
		DX_ASSERT( hr == S_OK, "" );
		hr = reflectionContainer->Load( objectBlob.get() );
		DX_ASSERT( hr == S_OK, "" );
		hr = reflectionContainer->FindFirstPartKind( MAKEFOURCC( 'D', 'X', 'I', 'L' ), &shaderIdx );
		DX_ASSERT( hr == S_OK, "" );

		/*
			* for simplicity,
			- NumDescriptors = 1
			- focus UAV, Constant

			ex) 
			RWStructuredBuffer<float> buffer0;
			RWStructuredBuffer<float> buffer1;
			RWStructuredBuffer<float> result;
			ConstantBuffer<Parameter> params;

			signature: 
			[0] CBV, "params"
			[1] UAV, "buffer0"
			[2] UAV, "buffer1"
			[3] UAV, "params"
			*/

		DxPtr<ID3D12ShaderReflection> reflection;
		hr = reflectionContainer->GetPartReflection( shaderIdx, IID_PPV_ARGS( reflection.getAddressOf() ) );
		DX_ASSERT( hr == S_OK, "" );

		// Use reflection interface here.
		D3D12_SHADER_DESC desc = {};
		reflection->GetDesc( &desc );

		std::vector<D3D12_DESCRIPTOR_RANGE> bufferDescriptorRanges;
		for( auto i = 0; i < desc.BoundResources; ++i )
		{
			D3D12_SHADER_INPUT_BIND_DESC bind = {};
			reflection->GetResourceBindingDesc( i, &bind );
			D3D12_DESCRIPTOR_RANGE range = {};
			switch( bind.Type )
			{
			case D3D_SIT_CBUFFER:
			{
				range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
				ID3D12ShaderReflectionConstantBuffer* cb = reflection->GetConstantBufferByName( bind.Name );
				D3D12_SHADER_BUFFER_DESC cbDesc;
				cb->GetDesc( &cbDesc );

				_cbv[bind.Name].location = i;
				_cbv[bind.Name].bytes = cbDesc.Size;
				break;
			}
			case D3D_SIT_UAV_RWTYPED:
			case D3D_SIT_UAV_RWSTRUCTURED:
			{
				range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
				_uav[bind.Name].location = i;
				_uav[bind.Name].stride = bind.NumSamples; // https://docs.microsoft.com/en-us/windows/win32/api/d3d12shader/ns-d3d12shader-d3d12_shader_input_bind_desc
				break;
			}
			case D3D_SIT_UAV_RWBYTEADDRESS:
			{
				range.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
				_uav[bind.Name].location = i;
				_uav[bind.Name].stride = 0; // RWByteAddressBuffer
				break;
			}
			default:
				DX_ASSERT( 0, "" );
			}

			range.NumDescriptors = 1;
			range.BaseShaderRegister = bind.BindPoint;
			range.RegisterSpace = bind.Space;
			range.OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
			bufferDescriptorRanges.push_back( range );
		}

		D3D12_ROOT_PARAMETER rootParameter = {};
		rootParameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
		rootParameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
		rootParameter.DescriptorTable.NumDescriptorRanges = bufferDescriptorRanges.size();
		rootParameter.DescriptorTable.pDescriptorRanges = bufferDescriptorRanges.data();

		// Signature
		D3D12_ROOT_SIGNATURE_DESC rsDesc = CD3DX12_ROOT_SIGNATURE_DESC( 1, &rootParameter );
		DxPtr<ID3DBlob> signatureBlob;
		hr = D3D12SerializeRootSignature( &rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, signatureBlob.getAddressOf(), nullptr );
		DX_ASSERT( hr == S_OK, "" );

		hr = device->d3d12Device()->CreateRootSignature( 0, signatureBlob->GetBufferPointer(), signatureBlob->GetBufferSize(), IID_PPV_ARGS( _signature.getAddressOf() ) );
		DX_ASSERT( hr == S_OK, "" );

		D3D12_COMPUTE_PIPELINE_STATE_DESC ppDesc = {};
		ppDesc.CS.pShaderBytecode = objectBlob->GetBufferPointer();
		ppDesc.CS.BytecodeLength = objectBlob->GetBufferSize();
		ppDesc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;
		ppDesc.NodeMask = 0;
		ppDesc.pRootSignature = _signature.get();
		hr = device->d3d12Device()->CreateComputePipelineState( &ppDesc, IID_PPV_ARGS( _csPipeline.getAddressOf() ) );
		DX_ASSERT( hr == S_OK, "" );
	}
	struct UAVType
	{
		int location;
		int stride;
	};
	struct CBVType
	{
		int location;
		int bytes;
	};
	class Argument
	{
	public:
		Argument( Device* device, std::map<std::string, Shader::UAVType> uav, std::map<std::string, Shader::CBVType> cbv )
			: _increment( 0 ), _uav( uav ), _cbv(), _device( device )
		{
			HRESULT hr;
			D3D12_DESCRIPTOR_HEAP_DESC desc = {};
			desc.NumDescriptors = uav.size() + cbv.size();
			desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
			desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
			hr = device->d3d12Device()->CreateDescriptorHeap( &desc, IID_PPV_ARGS( _bufferHeap.getAddressOf() ) );
			DX_ASSERT( hr == S_OK, "" );

			_increment = device->d3d12Device()->GetDescriptorHandleIncrementSize( D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV );

			for( auto constantBuffer : cbv )
			{
				// allocate constant buffer
				std::unique_ptr<ConstantBuffer> c( new ConstantBuffer( device, constantBuffer.second.bytes, string_to_wstring( "Arg-" + constantBuffer.first ).c_str() ) );

				// build buffer view.
				D3D12_CONSTANT_BUFFER_VIEW_DESC d = {};
				d.BufferLocation = c->d3d12Resource()->GetGPUVirtualAddress();
				d.SizeInBytes = c->bytes();
				D3D12_CPU_DESCRIPTOR_HANDLE h = _bufferHeap->GetCPUDescriptorHandleForHeapStart();
				h.ptr += (int64_t)_increment * constantBuffer.second.location;
				device->d3d12Device()->CreateConstantBufferView( &d, h );

				_cbv[constantBuffer.first] = std::move( c );
			}
		}
		void RWStructured( const char* var, Buffer* resource )
		{
			DX_ASSERT( _uav.count( var ), "" );

			D3D12_UNORDERED_ACCESS_VIEW_DESC d = {};
			d.Format = DXGI_FORMAT_UNKNOWN;
			d.Buffer.FirstElement = 0;
			d.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
			if( _uav[var].stride == 0 ) // for RWByteAddressBuffer
			{
				d.Format = DXGI_FORMAT_R32_TYPELESS;
				d.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
				d.Buffer.NumElements = resource->bytes() / 4;
			}
			else
			{
				d.Buffer.NumElements = resource->bytes() / _uav[var].stride;
				d.Buffer.StructureByteStride = _uav[var].stride;
			}
			
			d.Buffer.CounterOffsetInBytes = 0;
			D3D12_CPU_DESCRIPTOR_HANDLE h = _bufferHeap->GetCPUDescriptorHandleForHeapStart();
			h.ptr += (int64_t)_increment * _uav[var].location;

			_device->d3d12Device()->CreateUnorderedAccessView( resource->d3d12Resource(), nullptr, &d, h );
		}
		template <class T>
		void Constant( const char* var, T value )
		{
			DX_ASSERT( _cbv.count( var ), "" );
			DX_ASSERT( _cbv[var]->typeBytes() == sizeof( T ), "" );
			memcpy( _cbv[var]->ptr(), &value, sizeof( T ) );
		}
		ID3D12DescriptorHeap* d3d12DescriptorHeap()
		{
			return _bufferHeap.get();
		}

	private:
		uint32_t _increment;
		std::map<std::string, UAVType> _uav;
		std::map<std::string, std::unique_ptr<ConstantBuffer>> _cbv;
		DxPtr<ID3D12DescriptorHeap> _bufferHeap;
		Device* _device;
	};

	Argument* newArgument( Device* device ) const
	{
		return new Argument( device, _uav, _cbv );
	}
	void dispatchAsync( Device* device, Argument* arg, int64_t x, int64_t y, int64_t z )
	{
		device->enqueueCommand( [&]( ID3D12GraphicsCommandList* commandList )
								{
									ID3D12DescriptorHeap* heap = arg->d3d12DescriptorHeap();
									commandList->SetDescriptorHeaps( 1, &heap );
									commandList->SetPipelineState( _csPipeline.get() );
									commandList->SetComputeRootSignature( _signature.get() );
									commandList->SetComputeRootDescriptorTable( 0, heap->GetGPUDescriptorHandleForHeapStart() );
									commandList->Dispatch( x, y, z );
								} );
	}

private:
	DxPtr<ID3D12RootSignature> _signature;
	DxPtr<ID3D12PipelineState> _csPipeline;
	std::map<std::string, UAVType> _uav;
	std::map<std::string, CBVType> _cbv;
};
} // namespace dx